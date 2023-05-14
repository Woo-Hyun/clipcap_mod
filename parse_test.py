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
from sklearn.model_selection import train_test_split
from PIL import ImageDraw

def process_data(clip_model, preprocess, data, output_path, device):
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_crop_embeddings = []
    all_captions = []
    clip_embedding = 0 # for d["clip_embedding"]
    d = data[1]
    filename = d["img_path"]
    print(filename)
    if not os.path.isfile(filename):
        print("there is no img")
        print(filename)
    image = io.imread(filename)
    crop_image = image
    image = Image.fromarray(image)
    image.save('./image.png', 'png')
    #image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

    if len(d["geometry"]) != 0 and d["geometry"][2][1] - d["geometry"][0][1] != 0 and d["geometry"][2][0] - d["geometry"][0][0] != 0:
        crop_image = Image.fromarray(crop_image)
        crop_image = crop_image.crop((d["geometry"][0][0], d["geometry"][0][1],
                                      d["geometry"][2][0], d["geometry"][2][1]))
        crop_image.save('./crop.png', 'png')
        crop_image = preprocess(crop_image).unsqueeze(0).to(device)
    else:
        crop_image = preprocess(Image.fromarray(crop_image)).unsqueeze(0).to(device)
        
    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    train_out_path = f"./data/drama/woo_split_{clip_model_name}_crop_train_test.pkl"
    val_out_path = f"./data/drama/woo_split_{clip_model_name}_crop_val_test.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    with open('../drama_data/integrated_output_clipcap.json', 'r') as f:
        data = json.load(f)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    print("Processing train set...")
    process_data(clip_model, preprocess, train_data, train_out_path, device)

    #print("Processing validation set...")
    #process_data(clip_model, preprocess, val_data, val_out_path, device)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
