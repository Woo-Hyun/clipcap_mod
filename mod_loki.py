import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
import argparse
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
from PIL import Image
import json
import time
import re
from torchvision.transforms import ToPILImage
from PIL import ImageOps

def denormalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    mean = torch.Tensor(mean).unsqueeze(1).unsqueeze(2).to(tensor.device)
    std = torch.Tensor(std).unsqueeze(1).unsqueeze(2).to(tensor.device)
    return tensor * std + mean

def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class ClipCaptionDramaModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, 
                crop_prefix: torch.Tensor, bbox: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        crop_prefix = torch.cat((crop_prefix, bbox), dim=1)
        prefix = torch.cat((prefix, crop_prefix), dim=1)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        #mask = torch.cat((mask, mask), dim=0)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8):
        super(ClipCaptionDramaModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size * 2 + 6, (self.gpt_embedding_size * prefix_length) // 2,
                                 self.gpt_embedding_size * prefix_length))


def main(loki_dir: str):
    ##### load model #####

    model_path = "./drama_train/flip_bbox_center_norm/drama_prefix-009.pt"
    
    is_gpu = True #@param {type:"boolean"}
    use_beam_search = False #@param {type:"boolean"}

    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prefix_length = 10
    model = ClipCaptionDramaModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
    model = model.eval()
    model = model.to(device)

    for (root, dirs, files) in tqdm(list(os.walk(loki_dir)), position=0, desc='main'):
        # if len(dirs) > 0:
        #     for dir_name in dirs:
        #         print(root+dir_name)
        if len(files) > 0:
            for file in tqdm(files, position=1, desc='sub', leave=False):
                name, extension = os.path.splitext(file)
                if extension == ".json":

                    with open(root + '/' + name + extension, 'r') as f:
                        json_data = json.load(f)
                    
                    categories = ["Car", "Bus", "Truck", "Van", "Motorcyclist", "Bicyclist", "Pedestrian", "Wheelchair", "Traffic_Sign", "Traffic_Light"]
                    for category in categories:
                        for item in json_data[category]:

                            filename = root + '/image_' + name[-4:] + '.png'
                            if not os.path.isfile(filename):
                                print("there is no img")
                                print(filename)
                                continue

                            image = io.imread(filename)
                            pil_image = Image.fromarray(image)
                            w, h = pil_image.size

                            ###############
                            # 너비와 높이 중 더 긴 변의 길이를 찾습니다.
                            max_side = max(w, h)

                            # 패딩을 계산합니다.
                            left_padding = (max_side - w) // 2
                            right_padding = max_side - w - left_padding
                            top_padding = (max_side - h) // 2
                            bottom_padding = max_side - h - top_padding

                            # 패딩을 적용합니다.
                            padded_image = ImageOps.expand(pil_image, (left_padding, top_padding, right_padding, bottom_padding), fill=(0,0,0))

                            #image = preprocess(pil_image).unsqueeze(0).to(device)
                            image = preprocess(padded_image).unsqueeze(0).to(device)
                            image = denormalize(image)
                            to_pil = ToPILImage()
                            saved_image = to_pil(image.squeeze(0).cpu())
                            saved_image_path = "./example_de.png"
                            saved_image.save(saved_image_path)

                            exit()
                            ######################

                            if json_data[category][item]["box"]["width"] < 100 and json_data[category][item]["box"]["height"] < 100:
                                continue

                            pos_min_x = json_data[category][item]["box"]["left"]
                            pos_min_y = json_data[category][item]["box"]["top"]
                            pos_max_x = pos_min_x + json_data[category][item]["box"]["width"]
                            pos_max_y = pos_min_y + json_data[category][item]["box"]["height"]
                            bbox = [pos_min_x, pos_min_y, 
                                    pos_max_x, pos_max_y,
                                    (pos_min_x+pos_max_x)/2, 
                                    (pos_min_y+pos_max_y)/2]
                            bbox = [coord / w if i % 2 == 0 else coord / h for i, coord in enumerate(bbox)]
                            bbox = torch.tensor(bbox).unsqueeze(0).to(device)
                            
                            crop_pil_image = pil_image.crop((pos_min_x, pos_min_y, pos_max_x, pos_max_y))
                            crop_pil_image = preprocess(crop_pil_image).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                                crop_prefix = clip_model.encode_image(crop_pil_image).to(device, dtype=torch.float32)
                                crop_prefix = torch.cat((crop_prefix, bbox), dim=1)
                                prefix = torch.cat((prefix, crop_prefix), dim=1)
                                prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

                            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
                            json_data[category][item]["caption"] = generated_text_prefix
                    
                    with open(root + '/' + name + '_with_caption' + extension, 'w') as f:
                        json.dump(json_data, f, indent='\t')
                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loki_dir', default="./image_example/")
    args = parser.parse_args()
    exit(main(args.loki_dir))