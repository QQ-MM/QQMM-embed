from typing import Dict, Optional
import torch
import torch.nn as nn 
import torch.distributed as dist
from torch import Tensor
from transformers import PreTrainedModel

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

# EmbedModel for GradCache
class EmbedModel(nn.Module):
    def __init__(self,
                 encoder: PreTrainedModel,
                 embed_token_id: int = -1,
                 im_start_id: int = 151644,
                 im_end_id: int = 151645,
                 image_f_len: int = 1,
                 normalize: bool = True,
                 temperature: float = 0.02,
                 alpha: float = 1.0,
                 lm_loss_weight: float = 1.0,
                 mask_precontext: bool = False,
                 llava: bool = False
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.embed_token_id = embed_token_id
        self.llava = llava
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.mask_precontext = mask_precontext
        self.image_f_len = image_f_len
        self.normalize = normalize
        self.temperature = temperature
        self.alpha = alpha
        self.lm_loss_weight = lm_loss_weight
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def make_diy_mask(self, inputs):
        attention_mask = inputs['attention_mask']
        if len(attention_mask.shape) == 2:
            sequence_length = attention_mask.shape[1]
            target_length = attention_mask.shape[1]
            dtype = torch.bfloat16
            device = inputs['input_ids'].device
            min_dtype = torch.finfo(dtype).min
            cache_position = torch.arange(0, sequence_length, device=attention_mask.device)
            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                        attention_mask,
                        sequence_length=sequence_length,
                        target_length=target_length,
                        dtype=dtype,
                        device=device,
                        min_dtype=min_dtype,
                        cache_position=cache_position,
                        batch_size=attention_mask.shape[0],
                    )
        else:
            dtype = torch.bfloat16
            min_dtype = torch.finfo(dtype).min
        mask = inputs['input_ids'] == self.embed_token_id
        embed_index = torch.argmax(mask.float(), dim=1)
        embed_index[embed_index==0] = inputs['input_ids'].shape[1]
        embed_index = embed_index.view(-1, )
        mask = inputs['input_ids'] == self.im_start_id
        im_start_index_tmp = torch.argmax(mask.float(), dim=1).view(-1, 1)
        mask = torch.scatter(mask, dim=1, index=im_start_index_tmp, value=False)
        im_start_index = torch.argmax(mask.float(), dim=1).view(-1, )
        mask = inputs['input_ids'] == self.im_end_id
        im_end_index_tmp = torch.argmax(mask.float(), dim=1).view(-1, 1)
        mask = torch.scatter(mask, dim=1, index=im_end_index_tmp, value=False)
        im_end_index = torch.argmax(mask.float(), dim=1).view(-1, )
        for b in range(attention_mask.shape[0]):
            attention_mask[b, 0, embed_index[b]+1:, im_start_index[b]:im_end_index[b]+2] = min_dtype # <|im_start|>user\nxxxxx<|im_end|>\n
        inputs['attention_mask'] = attention_mask
    
    def encode(self, *args, **kwargs):
        if self.llava:
            kwargs['embed_token_id'] = self.embed_token_id
            kwargs['return_emb'] = True
        outputs = self.encoder(*args, **kwargs)
        return outputs
        
    def encode_input(self, inputs):
        if self.llava:
            inputs.pop('score', None)
            inputs.pop('record', None)
            inputs.pop('prompts_ids', None)
            inputs['embed_token_id'] = self.embed_token_id
            if self.mask_precontext:
                inputs['mask_precontext'] = True
                inputs['im_start_id'] = self.im_start_id
                inputs['im_end_id'] = self.im_end_id
            inputs['return_emb'] = True
            inputs['cal_loss'] = True
            outputs = self.encoder(**inputs, output_hidden_states=True)
            emb = outputs['emb']
        else:
            mask = inputs['input_ids'] == self.embed_token_id
            if 'labels' in inputs.keys():
                inputs['labels'][mask] = -100
            hidden_index = torch.argmax(mask.float(), dim=1)
            hidden_index[hidden_index==0] = inputs['input_ids'].shape[1]
            if self.mask_precontext:
                self.make_diy_mask(inputs)
            inputs.pop('score', None)
            inputs.pop('record', None)
            inputs.pop('prompts_ids', None)
            outputs = self.encoder(**inputs, output_hidden_states=True)
            hidden_states = outputs['hidden_states'][-1]
            hidden_states_mean = hidden_states.mean()
            hidden_states = hidden_states + 0.0 * hidden_states_mean
            hidden_states = torch.gather(hidden_states, dim=1, index=(hidden_index-1).view(hidden_index.shape[0], 1, 1).repeat(1, 1, hidden_states.shape[-1]))
            emb = hidden_states[:, 0, :].contiguous() # B, C
        if self.normalize:
            emb_norm = torch.norm(emb, p=2, dim=-1, keepdim=True)
            emb = emb / torch.clip(emb_norm, min=1e-7)
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        loss_lm = self.lm_loss_weight * outputs['loss'] if isinstance(outputs, dict) else self.lm_loss_weight * outputs[0]
        return emb, loss_lm

    def forward(self, qry: Optional[Dict[str, Tensor]] = None, tgt: Optional[Dict[str, Tensor]] = None):
        qry_reps, qry_loss_lm = self.encode_input(qry) if qry else (None, torch.tensor(torch.nan))  # (bsz_per_device, dim)
        tgt_reps, tgt_loss_lm = self.encode_input(tgt) if tgt else (None, torch.tensor(torch.nan)) # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            if torch.isnan(qry_loss_lm) and torch.isnan(tgt_loss_lm):
                loss_lm = None
            elif torch.isnan(qry_loss_lm):
                loss_lm = tgt_loss_lm
            elif torch.isnan(tgt_loss_lm):
                loss_lm = qry_loss_lm
            else:
                loss_lm = qry_loss_lm + tgt_loss_lm
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}, loss_lm

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
