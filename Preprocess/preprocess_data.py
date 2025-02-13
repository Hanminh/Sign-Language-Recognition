import cv2 as cv
import os
import numpy as np
import pandas as pd
import glob
import pdb
import re
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
ROOT_PATH = os.getenv("DATA_PATH")
IMG_SIZE = (256, 256)

def csv2dict(dataset_type):
    anno_path = os.path.join(ROOT_PATH ,f"annotations\\manual\\PHOENIX-2014-T.{dataset_type}.corpus.csv") 
    print(anno_path)
    inputs_list = pd.read_csv(anno_path)
    inputs_list = (inputs_list.to_dict()['name|video|start|end|speaker|orth|translation'].values())
    info_dict = dict()
    info_dict['prefix'] = ROOT_PATH + "\\features\\fullFrame-210x260px"
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list), total=len(inputs_list)):
        name, video, start, end, speaker, orth, translation = file_info.split("|")
        num_frames = len(glob.glob(f"{info_dict['prefix']}\\{dataset_type}\\{name}\\*.png"))
        info_dict[file_idx] = {
            'fileid': name,
            'folder': f"{dataset_type}\\{name}",
            'signer': speaker,
            'label': orth,
            'num_frames': num_frames
        }
    return info_dict

def resize_image(image_path, dsize = IMG_SIZE):
    img = cv.imread(image_path)
    img = cv.resize(img, dsize, interpolation= cv.INTER_LANCZOS4)
    return img

def resize_dataset(video_idx, infor_dict, dsize = IMG_SIZE):
    prefix = infor_dict['prefix']
    info = infor_dict[video_idx]
    img_list = glob.glob(f"{prefix}\\{info['folder']}\\*.png")
    for img_path in tqdm(img_list):
        img = resize_image(img_path, dsize)
        rs_img_path = img_path.replace("210x260px", "256x256px")
        rs_img_path = rs_img_path.replace('png', 'jpg')
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv.imwrite(rs_img_path, img)
        else :
            cv.imwrite(rs_img_path, img)

def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split()
        for gloss in split_label:
            if gloss in total_dict.keys():
                total_dict[gloss] += 1
            else:
                total_dict[gloss] = 1
    return total_dict


def preprocess_data(dataset_type, gloss_dict = None):
    if gloss_dict is None:
        gloss_dict = dict()
    info_dict = csv2dict(dataset_type)
    # save the information dict as pickle file
    save_path = f"Information_dict\\{dataset_type}_info.pkl"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pd.to_pickle(info_dict, save_path)
    for video_idx in range(len(info_dict) - 1):
        resize_dataset(video_idx, info_dict)
    gloss_dict = sign_dict_update(gloss_dict, info_dict)
    return gloss_dict
        
if __name__ == "__main__":
    gloss_dict = dict()
    gloss_dict = preprocess_data('train', gloss_dict)
    gloss_dict = preprocess_data('dev', gloss_dict)
    gloss_dict = preprocess_data('test', gloss_dict)
    gloss_dict = sorted(gloss_dict.items(), key=lambda x: x[0])
    save_dict = {}
    for idx, (k,v) in enumerate(gloss_dict):
        save_dict[k] = [idx + 1, v]
    np.save("Information_dict\\gloss_dict.npy", save_dict)
    # save the gloss_dict as pickle file
    save_path = "Information_dict\\gloss_dict.pkl"
    pd.to_pickle(gloss_dict, save_path)
    
    print("Preprocess data done")