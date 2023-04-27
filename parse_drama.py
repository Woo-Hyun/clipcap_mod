import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import torchvision.transforms.functional as func
import numpy as np


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/drama/oscar_split_{clip_model_name}_crop.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('../drama_data/integrated_output_clipcap.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_crop_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        filename = d["img_path"]
        if not os.path.isfile(filename):
            print("there is no img")
            print(filename)
            continue
        image = io.imread(filename)
        crop_image = image
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

        if len(d["geometry"]) != 0 and d["geometry"][2][1] - d["geometry"][0][1] != 0 and d["geometry"][1][0] - d["geometry"][0][0] != 0:
            crop_image = Image.fromarray(crop_image)
            crop_image = func.crop(crop_image, d["geometry"][0][1], d["geometry"][0][0],
                                   d["geometry"][2][1] - d["geometry"][0][1],
                                   d["geometry"][1][0] - d["geometry"][0][0])
            crop_image = preprocess(crop_image).unsqueeze(0).to(device)
        else:
            crop_image = preprocess(Image.fromarray(crop_image)).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
            crop_prefix = clip_model.encode_image(crop_image).cpu()
        d["clip_embedding"] = i
        d["clip_crop_embedding"] = i
        d["image_id"] = i+1000
        all_embeddings.append(prefix)
        all_crop_embeddings.append(crop_prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), 
                             "clip_crop_embedding": torch.cat(all_crop_embeddings, dim=0), 
                             "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), 
                     "clip_crop_embedding": torch.cat(all_crop_embeddings, dim=0), 
                     "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))