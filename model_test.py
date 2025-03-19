import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data_loader_vn
import os
import torch
from Modules import BiLSTM
from Modules.BiLSTM import BiLSTM
from slr_network import SLR_Network
from argument import *
from tqdm import tqdm
import jiwer
import cv2 as cv
from Generate_Data import data_augmentation
from Generate_Data.data_augmentation import *

IMG_SIZE = (256, 256)
VIDEO_PATH = '00693_bạn gái.mp4'

path = ''
vn_id2gloss = np.load(path + 'Information_dict/vn_id2gloss.npy', allow_pickle=True).item()
vn_dictionary = np.load(path + 'Information_dict/vn_dictionary.npy', allow_pickle=True).item()
dictionary = []
dictionary.append(' ')
for i in list(vn_dictionary.keys()):
    dictionary.append(i + '|')
    
from pyctcdecode import build_ctcdecoder

import jiwer
def calculate_wer(pred, true):
    pred_words = pred.split('|')[:-1]
    pred_str = ' '.join(pred_words)
    wer_score = jiwer.wer(true, pred_str)
    return wer_score

def resize_image(img, dsize = IMG_SIZE):
    img = cv.resize(img, dsize, interpolation= cv.INTER_LANCZOS4)
    return img

def read_video(video_path):
    cap = cv.VideoCapture(video_path)
    video_data = []
    num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, 220:1060]
        frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_LANCZOS4)
        if num % 5 == 0:
            frame = resize_image(frame)
            video_data.append(frame)
        num += 1
    # print(num)
    video_data = np.array(video_data)
    input_data, *arg = normalize(video_data)
    cap.release()
    cv.destroyAllWindows()
    return input_data

def transform(transform_mode= False, input_size= (224, 224), image_scale= 1):
    if transform_mode:
        print("Apply training transform")
        return data_augmentation.Compose([
            # video_augmentation.CenterCrop(224),
            # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
            data_augmentation.RandomCrop(input_size),
            data_augmentation.RandomHorizontalFlip(0.5),
            data_augmentation.Resize(image_scale),
            data_augmentation.ToTensor()
        ])
    else:
        print("Apply validation transform")
        return data_augmentation.Compose([
            data_augmentation.CenterCrop(input_size),
            data_augmentation.Resize(image_scale),
            data_augmentation.ToTensor(),
        ])
        
def normalize(video, label= None, file_id= None):
    video, label = transform()(video, label, file_id)
    video = video.float() / 127.5 - 1
    return video, label
def predict_sentence(video_path= VIDEO_PATH):
    
    left_pad = 0
    last_stride = 1
    total_stride = 1
    kernel_sizes = ['K5', "P2", 'K5', "P2"]
    for layer_idx, ks in enumerate(kernel_sizes):
        if ks[0] == 'K':
            left_pad = left_pad * last_stride 
            left_pad += int((int(ks[1])-1)/2)
        elif ks[0] == 'P':
            last_stride = int(ks[1])
            total_stride = total_stride * last_stride


    input_data = read_video(video_path)
    input_data = input_data.unsqueeze(0)

    max_len = input_data.size(1)
    video_length = torch.LongTensor([np.ceil(input_data.size(1) / total_stride) * total_stride + 2*left_pad ])
    right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    max_len = max_len + left_pad + right_pad

    input_data = torch.cat(
        (
            input_data[0,0][None].expand(left_pad, -1, -1, -1),
            input_data[0],
            input_data[0,-1][None].expand(max_len - input_data.size(1) - left_pad, -1, -1, -1),
        )
        , dim=0).unsqueeze(0)

    model = SLR_Network(dictionary= dictionary, num_classes= len(dictionary) + 1 )
    model.to('cuda')
    input_data = input_data.to('cuda')
    output_model = model(input_data, torch.tensor([input_data.shape[1]]))
    return output_model['predictions']

# print(predict_sentence(video_path= VIDEO_PATH))