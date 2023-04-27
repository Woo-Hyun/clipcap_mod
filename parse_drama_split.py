import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

def process_data(clip_model, preprocess, data, output_path, device):
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        filename = d["img_path"]
        if not os.path.isfile(filename):
            print("there is no img")
            print(filename)
            continue
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        d["image_id"] = i+1000
        all_embeddings.append(prefix)
        all_captions.append(d)

    with open(output_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    train_out_path = f"./data/drama/woo_split_{clip_model_name}_train.pkl"
    val_out_path = f"./data/drama/woo_split_{clip_model_name}_val.pkl"
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
