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

def process_data(clip_model, preprocess, data, output_path, device):
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_crop_embeddings = []
    all_captions = []
    clip_embedding = 0 # for d["clip_embedding"]
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
            continue
            crop_image = preprocess(Image.fromarray(crop_image)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
            crop_prefix = clip_model.encode_image(crop_image).cpu()
        d["clip_embedding"] = clip_embedding # instead i
        d["clip_crop_embedding"] = clip_embedding
        clip_embedding += 1
        d["image_id"] = i+1000
        all_embeddings.append(prefix)
        all_crop_embeddings.append(crop_prefix)
        all_captions.append(d)

    with open(output_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), 
                     "clip_crop_embedding": torch.cat(all_crop_embeddings, dim=0), 
                     "captions": all_captions}, f)
        
    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    train_out_path = f"./data/drama/woo_split_{clip_model_name}_crop_train_2.pkl"
    val_out_path = f"./data/drama/woo_split_{clip_model_name}_crop_val_2.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    with open('../drama_data/integrated_output_clipcap.json', 'r') as f:
        data = json.load(f)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    print("Processing train set...")
    process_data(clip_model, preprocess, train_data, train_out_path, device)

    print("Processing validation set...")
    process_data(clip_model, preprocess, val_data, val_out_path, device)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
