import sys
import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data.dataloader import Dataset
from tqdm import tqdm

PROJECT_PATH = os.path.abspath('./')
sys.path.append(PROJECT_PATH)

from qqmm.models import build_processor
from qqmm.utils.parameter_manage import Parameters
from qqmm.models.qqmm_nav_qwen2.modeling_qqmm import QQMMForCausalLM
from qqmm.utils.chat import EmbedBot

def compute_recall(mapping, embed):
    N = len(mapping.keys())
    cnt = 0
    for src in mapping.keys():
        qry = embed[src]
        pos = embed[mapping[src]['pos'][0]]
        negs = []
        for tgt in mapping[src]['neg']:
            negs.append(embed[tgt])
        negs = torch.cat(negs, 0)
        p_sim = (qry * pos).sum()
        neg_sim = torch.matmul(qry, negs.transpose(0, 1)).max()
        if p_sim >= neg_sim:
            cnt += 1
    recall = float(cnt / N) * 100.
    return recall


class MMEBDataset(Dataset):
    def __init__(self, data_path, tgt_prompt="", prompt_type=None, img_size=None, llave=False):
        super().__init__()
        self.llave = llave
        meta = pd.read_parquet(data_path)
        self.data = []
        visited = {}
        self.mapping = {}
        cnt = 0
        self.img_size = img_size
        if prompt_type is None:
            prompt_type = 'default'
        for i in meta.index:
            qry_text = meta['qry_text'][i]
            qry_text = qry_text.replace('<|image_1|>\n', '')
            if str(meta['qry_img_path'][i]):
                qry_image_path = os.path.join(os.path.dirname(data_path), '../eval_images', str(meta['qry_img_path'][i]))
            else:
                qry_image_path = ''
            tmp = {'text': qry_text, 'image_path': qry_image_path, 'prompt': '', 'think': ''}
            src_cnt = cnt
            self.mapping[src_cnt] = {'pos': [], 'neg': []}
            self.data.append(tmp)
            visited[tmp['text'] + tmp['image_path']] = cnt
            cnt += 1
            tgt_text_list = meta['tgt_text'][i]
            tgt_image_path_list = meta['tgt_img_path'][i]
            _first = True
            for tgt_text, tgt_image_path in zip(tgt_text_list, tgt_image_path_list):
                tgt_text = tgt_text.replace('<|image_1|>\n', '')
                tgt_text = tgt_prompt + tgt_text if tgt_prompt else tgt_text
                if tgt_image_path:
                    tgt_image_path = os.path.join(os.path.dirname(data_path), '../eval_images', tgt_image_path)
                else:
                    tgt_image_path = ''
                if _first:
                    tmp = {'text': tgt_text, 'image_path': tgt_image_path, 'prompt': '', 'think': ''}
                    if tmp['text'] + tmp['image_path'] not in visited.keys():
                        self.data.append(tmp)
                        visited[tmp['text'] + tmp['image_path']] = cnt
                        cnt += 1
                    self.mapping[src_cnt]['pos'].append(visited[tmp['text'] + tmp['image_path']])
                    _first = False
                else:
                    tmp = {'text': tgt_text, 'image_path': tgt_image_path, 'prompt': '', 'think': ''}
                    if tmp['text'] + tmp['image_path'] not in visited.keys():
                        self.data.append(tmp)
                        visited[tmp['text'] + tmp['image_path']] = cnt
                        cnt += 1
                    self.mapping[src_cnt]['neg'].append(visited[tmp['text'] + tmp['image_path']])
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        data = self.data[index]
        text = data.get('text')
        image_path = data.get('image_path')
        prompt = data.get('prompt', '')
        think = data.get('think', '')
        img = Image.open(image_path).convert('RGB') if image_path else None
        # if img:
        #     w, h = img.size
        #     if min(w, h) < 32:
        #         img = None
        #         image_path = ''
        if img and self.img_size:
            w, h = img.size
            if w >= h:
                img = img.resize((self.img_size, int(self.img_size/w*h)))
            else:
                img = img.resize((int(self.img_size/h*w), self.img_size))
        # if img:
        #     w, h = img.size
        #     if max(w, h) > 2304:
        #         if w >= h:
        #             img = img.resize((2304, int(2304/w*h)))
        #         else:
        #             img = img.resize((int(2304/h*w), 2304))
        record = data.copy()
        record['text'] = text
        record['image'] = [img] if img else []
        # record['prompt'] = prompt
        # record['think'] = think
        if not self.llave:
            pass
        else:
            record['image'] = img
        # record = prepare_message_rl(record)
        return record     

config = Parameters()
config.merge_from_yaml('./configs/embed/qqmm-embed/mmeb.yaml')

print(">>> Building Model...")
processor = build_processor(config.PROCESSOR_CONFIG, inferring=True)
model = QQMMForCausalLM.from_pretrained('/group/40048/youzexue/shares/models/mllm/models/qqmm-v305f/release/v1', torch_dtype=torch.bfloat16, device_map='cuda')
bot = EmbedBot(model, processor)

print(">>> Loading Data...")
dataset = MMEBDataset('/group/40048/youzexue/QQMM/datasets/MMEB/MMEB-eval/Place365/test-00000-of-00001.parquet', tgt_prompt='Represent the following answer to an image classification task: ')

print(">>> Evaluating...")
embed = []
total_length = len(dataset)
for i in tqdm(range(total_length)):
    record = dataset.__getitem__(i)
    feat = bot.chat(text=record['text'], image=record['image'])
    embed.append(feat)
recall = compute_recall(dataset.mapping, embed)
print('Place365: ', recall)