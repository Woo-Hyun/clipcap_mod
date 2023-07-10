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
from IPython.display import display
import re

def process_data(clip_model, preprocess, data, output_path, device):
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_prev_embeddings = []
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

        # loading previous frame
        pre, frame_num_str, png = re.match(r"(.*frame_)(\d+)(\.png)", filename).groups()
        frame_num = int(frame_num_str)
        if "titan" in pre:
            prev_frame_num = max(0, frame_num - 12)
        else:
            prev_frame_num = max(0, frame_num - 9)
        prev_frame_filename = f"{pre}{prev_frame_num:06d}.png"
        if os.path.isfile(prev_frame_filename):
            prev_image = io.imread(prev_frame_filename)
            prev_image = Image.fromarray(prev_image)
        else:
            print(f"there is no previous img for {filename}")
            continue

        image = io.imread(filename)
        image = Image.fromarray(image)
        w, h = image.size

        # flip image
        crop_image = image
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        crop_flipped_image = None
        prev_flipped_image = prev_image.transpose(Image.FLIP_LEFT_RIGHT)

        # unsqeueeze(0) for batch dimension
        image = preprocess(image).unsqueeze(0).to(device)
        flipped_image = preprocess(flipped_image).unsqueeze(0).to(device)
        prev_image = preprocess(prev_image).unsqueeze(0).to(device)
        prev_flipped_image = preprocess(prev_flipped_image).unsqueeze(0).to(device)

        if len(d["geometry"]) != 0 and d["geometry"][2][1] - d["geometry"][0][1] != 0 and d["geometry"][2][0] - d["geometry"][0][0] != 0:
            if d["geometry"][2][1] - d["geometry"][0][1] < 100 or d["geometry"][2][0] - d["geometry"][0][0] < 100:
                continue
            crop_image = crop_image.crop((d["geometry"][0][0], d["geometry"][0][1],
                                        d["geometry"][2][0], d["geometry"][2][1]))
            crop_flipped_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)

            crop_image = preprocess(crop_image).unsqueeze(0).to(device)
            crop_flipped_image = preprocess(crop_flipped_image).unsqueeze(0).to(device)
        else:
            continue
        
        # get clip embedding with encode_image
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
            flipped_prefix = clip_model.encode_image(flipped_image).cpu()
            crop_prefix = clip_model.encode_image(crop_image).cpu()
            crop_flipped_prefix = clip_model.encode_image(crop_flipped_image).cpu()
            prev_prefix = clip_model.encode_image(prev_image).cpu()
            prev_flipped_prefix = clip_model.encode_image(prev_flipped_image).cpu()

        d["clip_embedding"] = clip_embedding # instead i
        d["clip_prev_embedding"] = clip_embedding
        d["clip_crop_embedding"] = clip_embedding
        clip_embedding += 1
        d["image_id"] = i+1000
        d["img_w"] = w
        d["img_h"] = h
        all_embeddings.append(prefix)
        all_prev_embeddings.append(prev_prefix)
        all_crop_embeddings.append(crop_prefix)
        all_captions.append(d)

        d["clip_embedding"] = clip_embedding # instead i
        d["clip_prev_embedding"] = clip_embedding
        d["clip_crop_embedding"] = clip_embedding
        clip_embedding += 1
        d["image_id"] = i+2000
        d["img_w"] = w
        d["img_h"] = h
        all_embeddings.append(flipped_prefix)
        all_prev_embeddings.append(prev_flipped_prefix)
        all_crop_embeddings.append(crop_flipped_prefix)
        d["geometry"][0][0], d["geometry"][2][0] = w - d["geometry"][2][0], w - d["geometry"][0][0]
        d["geometry"][1][0], d["geometry"][3][0] = w - d["geometry"][3][0], w - d["geometry"][1][0]
        temp_str = "TEMP_STRING"
        if 'left' in d["caption"]:
            d["caption"] = d["caption"].replace('left', temp_str)
        if 'right' in d["caption"]:
            d["caption"] = d["caption"].replace('right', 'left')
        if temp_str in d["caption"]:
            d["caption"] = d["caption"].replace(temp_str, 'right')
        all_captions.append(d)

    with open(output_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), 
                     "clip_prev_embedding": torch.cat(all_prev_embeddings, dim=0),
                     "clip_crop_embedding": torch.cat(all_crop_embeddings, dim=0), 
                     "captions": all_captions}, f)
        
    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    train_out_path = f"./data/drama/woo_split_crop_flip_100_wh_prev_train.pkl"
    val_out_path = f"./data/drama/woo_split_crop_flip_100_wh_prev_val.pkl"
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
