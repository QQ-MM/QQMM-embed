from typing import Optional, List
import torch
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM

from .configuration_qqmm import QQMMConfig
from .modeling_abstractor import PerceiverProjection


def build_vision_model(config, model=None):
    if model is None:
        model = AutoModel.from_config(config, trust_remote_code=True)

    model_type = model.config.model_type
    assert 'navit' in model_type, "Only support navit vision models."

    return model


class QQMMPreTrainedModel(PreTrainedModel):
    config_class = QQMMConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True


class QQMMForCausalLM(QQMMPreTrainedModel):
    def __init__(self,
                 config: QQMMConfig,
                 language_model=None,
                 vision_model=None):
        super().__init__(config)

        vision_model = build_vision_model(config.vision_model_config, vision_model)

        vision_abstractor = PerceiverProjection(**config.vision_abstractor_config,
                                                in_dim=self.config.vision_model_config.hidden_size,
                                                out_dim=self.config.language_model_config.hidden_size)

        if language_model is None:
            kwargs_ = {}
            if config._attn_implementation_internal is not None:
                kwargs_['attn_implementation'] = config._attn_implementation_internal
            language_model = AutoModelForCausalLM.from_config(config.language_model_config, trust_remote_code=True,
                                                              **kwargs_)

        self.vision_model = vision_model
        self.vision_abstractor = vision_abstractor
        self.language_model = language_model

        self.vision_output_key = parse_output_key(self.config.vision_output_key)

    def forward_vision(self, pixel_values, grid_sizes):
        is_dummy_input = pixel_values.size(0) == 0
        if is_dummy_input:
            pixel_values = torch.zeros((4,) + pixel_values.shape[1:],
                                       dtype=pixel_values.dtype, device=pixel_values.device)
            grid_sizes = torch.full((1,) + grid_sizes.shape[1:], fill_value=2,
                                    dtype=grid_sizes.dtype, device=grid_sizes.device)

        outputs = self.vision_model(pixel_values, grid_sizes)

        for k in self.vision_output_key:
            outputs = outputs[k]
        vision_embeds = outputs

        if is_dummy_input:
            vision_embeds = vision_embeds[:0]
            grid_sizes = grid_sizes[:0]

        vision_embeds = self.vision_abstractor(vision_embeds, grid_sizes)

        return vision_embeds

    def prepare_for_lm(self, input_ids, vision_embeds):
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if vision_embeds is not None:
            vision_mask = input_ids == self.config.image_token_id
            # assert torch.count_nonzero(vision_mask).item() == vision_embeds.shape[:-1].numel(), \
            #     "vision embeddings mismatch input embeddings: " \
            #     f"vision_mask shape={vision_mask.shape}; " \
            #     f"vision_mask count={torch.count_nonzero(vision_mask)}; " \
            #     f"vision_embeds shape={vision_embeds.shape}"
            inputs_embeds = torch.masked_scatter(inputs_embeds, vision_mask.unsqueeze(-1),
                                                 vision_embeds.to(inputs_embeds.dtype).view(-1, vision_embeds.size(-1)))

        return {'inputs_embeds': inputs_embeds}

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.BoolTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                *,
                pixel_values: Optional[torch.Tensor] = None,
                grid_sizes: Optional[torch.Tensor] = None,
                vision_embeds: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: bool = True,
                **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ========Get visual embedding========
        if pixel_values is not None and vision_embeds is None:
            vision_embeds = self.forward_vision(pixel_values, grid_sizes)
        # if self._gradient_checkpointing and vision_embeds is not None and not vision_embeds.requires_grad:
        #     vision_embeds.requires_grad_(True)

        # ========Prepare inputs for LM========
        kwargs_ = self.prepare_for_lm(input_ids, vision_embeds)
        kwargs.update(kwargs_)
        inputs_embeds = kwargs.pop('inputs_embeds')

        if self.is_gradient_checkpointing and torch.is_grad_enabled():
            inputs_embeds.requires_grad_(True)

        # ========Forward into LM========
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs)

        # ========Compute Loss========
        if labels is not None:
            logits = outputs['logits'] if return_dict else outputs[0]
            loss = self.loss_function(logits=logits, labels=labels)
            if return_dict:
                outputs['loss'] = loss
            else:
                outputs = (loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      cache_position=None,
                                      position_ids=None,
                                      use_cache=True,
                                      *,
                                      pixel_values: Optional[torch.Tensor] = None,
                                      grid_sizes: Optional[torch.Tensor] = None,
                                      vision_embeds: Optional[torch.FloatTensor] = None,
                                      **kwargs):
        cur_position = cache_position[0].item()
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = (position_ids[:, -(attention_mask.size(1) - cur_position):]
                            + (attention_mask.size(1) - position_ids.size(1)))

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        if cur_position == 0:
            model_inputs['pixel_values'] = pixel_values
            model_inputs['grid_sizes'] = grid_sizes
            model_inputs['vision_embeds'] = vision_embeds

        return model_inputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.language_model.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()


def parse_output_key(output_key):
    output_key_ = []
    if not output_key:
        return output_key_
    ks = [k.strip() for k in output_key.split(',')]
    i = 0
    while i < len(ks):
        k = ks[i]
        if k.startswith('['):
            for j in range(len(ks)-1, i, -1):
                if ks[j].endswith(']'):
                    break
            else:
                raise ValueError(output_key)
            ns = tuple(parse_output_key(','.join([k.lstrip('[')] + ks[i+1:j] + [ks[j].rstrip(']')])))
            output_key_.append(ns)
            i = j
        elif k.lstrip('-').isdigit():
            output_key_.append(int(k))
        elif ':' in k:
            ns = [int(n) if n != '' else None for n in k.split(':')]
            output_key_.append(slice(*ns))
        else:
            output_key_.append(k)
        i += 1

    return output_key_
