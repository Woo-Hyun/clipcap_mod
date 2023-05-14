import os
import glob
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
import json
import random
import pandas as pd
from itertools import permutations
# import utils
import matplotlib.pyplot as plt
import open3d as o3d


def make_dataset(images: list, label3d: list, label2d: list, odom: list,seq_id: int):
    df_label3d = []
    # with open(label2d[0], 'r') as f:
    #     data = json.load(f)
    for i, (image, label_3d, label_2d, odometry) in enumerate(zip(images, label3d, label2d, odom)):
        df = pd.read_csv(label_3d, delimiter=',', header=None, names=['labels', 'TRACK_ID', 'stationary', 'pos_x', 'pos_y', 'pos_z', 'dim_x', 'dim_y', 'dim_z', 'yaw', 'vehicle_state', 'intended_actions', 'potential_destination', 'direction','lane_information','box','age','gender','lidar','traffic_light_state','traffic_sign_type'])
        df = df.drop(0)
        df_odom = pd.read_csv(odometry, delimiter=',', header=None, names=['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw'])
        df_odom['labels'] = 'Car'
        df_odom['TRACK_ID'] = 'AV'
        df_odom['stationary'] = 'dynamic'
        df=pd.concat([df, df_odom]).reset_index(drop=True)
        with open(label_2d, 'r') as f:
            data = json.load(f)
        for key in data.keys():
            for track_id in data[key].keys():
                if key != 'Potential_Destination':
                    index=df[df['TRACK_ID']==track_id].index
                    if key == 'Pedestrian' or key =='Wheelchair':
                        if data[key][track_id]['not_in_lidar']==True or len(index)==0:
                            new_row = pd.Series({
                                'labels': key,
                                'TRACK_ID': track_id,
                                'box': [data[key][track_id]['box']['top'],data[key][track_id]['box']['left'],data[key][track_id]['box']['height'],data[key][track_id]['box']['width']],
                                'age': data[key][track_id]['attributes']['age'],
                                'gender': data[key][track_id]['attributes']['gender'],
                                'lidar': True
                            })
                            # df=pd.concat([df, new_row.to_frame().T], ignore_index=True)
                        else:
                            index=index[0]
                            new_row = df.loc[index].copy()
                            new_row['box'] = [data[key][track_id]['box']['top'],data[key][track_id]['box']['left'],data[key][track_id]['box']['height'],data[key][track_id]['box']['width']]
                            new_row['age'] = data[key][track_id]['attributes']['age']
                            new_row['gender'] = data[key][track_id]['attributes']['gender']
                            new_row['lidar'] = False
                            df=pd.concat([df, new_row.to_frame().T], ignore_index=True).reset_index(drop=True)
                            df=df.drop(index).reset_index(drop=True)

                    elif key == 'Traffic_Light':
                        new_row = pd.Series({
                                'labels': key,
                                'TRACK_ID': track_id,
                                'box': [data[key][track_id]['box']['top'],data[key][track_id]['box']['left'],data[key][track_id]['box']['height'],data[key][track_id]['box']['width']],
                                'traffic_light_state': data[key][track_id]['attributes']['traffic_light_state'],
                                'lidar': True
                            })
                        df=pd.concat([df, new_row.to_frame().T], ignore_index=True).reset_index(drop=True)
                    elif key == 'Traffic_Sign':
                        new_row = pd.Series({
                                'labels': key,
                                'TRACK_ID': track_id,
                                'box': [data[key][track_id]['box']['top'],data[key][track_id]['box']['left'],data[key][track_id]['box']['height'],data[key][track_id]['box']['width']],
                                'traffic_sign_type': data[key][track_id]['attributes']['traffic_sign_type'],
                                'lidar': True
                            })
                        df=pd.concat([df, new_row.to_frame().T], ignore_index=True).reset_index(drop=True)
                    elif key == 'Road_Entrance_Exit':
                        if data[key][track_id]['not_in_lidar']==True or len(index)==0:
                            new_row = pd.Series({
                                'labels': key,
                                'TRACK_ID': track_id,
                                'box': [data[key][track_id]['box']['top'],data[key][track_id]['box']['left'],data[key][track_id]['box']['height'],data[key][track_id]['box']['width']],
                                'lidar': True
                            })
                            df=pd.concat([df, new_row.to_frame().T], ignore_index=True).reset_index(drop=True)
                        else:
                            index=index[0]
                            new_row = df.loc[index].copy()
                            new_row['box'] = [data[key][track_id]['box']['top'],data[key][track_id]['box']['left'],data[key][track_id]['box']['height'],data[key][track_id]['box']['width']]
                            new_row['lidar'] = False
                            df=pd.concat([df, new_row.to_frame().T], ignore_index=True).reset_index(drop=True)
                            df=df.drop(index).reset_index(drop=True)
                    else:
                        if data[key][track_id]['not_in_lidar']==True or len(index)==0:
                            new_row = pd.Series({
                                'labels': key,
                                'TRACK_ID': track_id,
                                'box': [data[key][track_id]['box']['top'],data[key][track_id]['box']['left'],data[key][track_id]['box']['height'],data[key][track_id]['box']['width']],
                                'lane_information': data[key][track_id]['attributes']['lane_information'],
                                'lidar': True
                            })
                            # df=pd.concat([df, new_row.to_frame().T], ignore_index=True).reset_index(drop=True)
                        else:
                            index=index[0]
                            new_row = df.loc[index].copy()
                            new_row['box'] = [data[key][track_id]['box']['top'],data[key][track_id]['box']['left'],data[key][track_id]['box']['height'],data[key][track_id]['box']['width']]
                            new_row['lane_information'] = data[key][track_id]['attributes']['lane_information']
                            new_row['lidar'] = False
                            df=pd.concat([df, new_row.to_frame().T], ignore_index=True).reset_index(drop=True)
                            df=df.drop(index).reset_index(drop=True)
        # reset index
        df['TIMESTAMP']=i
        df['seq_id']=seq_id
        df['image_path']=image
        df['lane_information']=df['lane_information'].fillna(0)
        df['vehicle_state']=df['vehicle_state'].fillna(0)
        df_label3d.append(df)
    return pd.concat(df_label3d, ignore_index=True).reset_index(drop=True)
label3d=0
dict_data=0
data=0
train_num=0
val_num=0
train_path = '../data/train/data/'
val_path = '../data/val/data/'
random_numbers = random.sample(range(0, 645), 120)
for i in tqdm(range(len(folder))):
    images = sorted(glob.glob(folder[i] + '/image_*.png'))
    label3d = sorted(glob.glob(folder[i] + '/label3d*'))
    label2d = sorted(glob.glob(folder[i] + '/label2d*'))
    odom = sorted(glob.glob(folder[i] + '/odom*'))
    if len(images) < 40:
        continue
    else:
        for j in range(int(len(images)/40)):
            data=make_dataset(images[40*j:40*(j+1)],label3d[40*j:40*(j+1)],label2d[40*j:40*(j+1)],odom[40*j:40*(j+1)],i)
            if i in random_numbers:
                data.to_csv(val_path + str(val_num) + '.csv', index=False)
                val_num+=1
            else:
                data.to_csv(train_path + str(train_num) + '.csv', index=False)
                train_num+=1

            
            