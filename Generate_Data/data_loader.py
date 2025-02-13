from Generate_Data import data_augmentation
from Generate_Data.data_augmentation import *
import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
import numpy as np
import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler
import pickle
from dotenv import load_dotenv
load_dotenv()

class VideoDataset(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, transform_mode= True, frame_interval= 1, image_scale= 1.0, kernel_size= 1, input_size= 224, mode= 'train', data_type= 'video'):
        self.mode= mode
        self.prefix= prefix
        self.transform_mode= True if mode == 'train' else False
        self.image_scale= image_scale
        global kernel_sizes
        kernel_sizes= kernel_size
        self.data_type= data_type
        self.dict= gloss_dict
        self.num_gloss= num_gloss
        self.input_size= input_size
        self.frame_interval= frame_interval
        self.drop_ratio= drop_ratio
        self.feature_prefix = f"{self.prefix}\\features\\fullFrame-256x256px\\{self.mode}"
        # load the pickle file
        self.inputs_list= pickle.load(open(f'Information_dict\\{self.mode}_info.pkl', 'rb'))
        self.data_aug = self.transform()
        
    def __getitem__(self, index):
        if self.data_type == 'video':
            input_data, label, file_info = self.read_video(index)
            input_data, label = self.normalize(input_data, label, file_info)
            return input_data, torch.LongTensor(label), self.inputs_list[index]['label']
        elif self.data_type == 'feature':
            input_data, label = self.read_feature(index)
            return input_data, label, self.inputs_list[index]['label']
    
    def read_video(self, index):
        # load file info
        file_info =self.inputs_list[index]
        img_folder = self.prefix + f'\\features\\fullFrame-256x256px\\{self.mode}\\' + file_info['fileid']
        # print(img_folder)
        img_list = sorted(glob.glob(img_folder + '\\*.jpg'))
        # print(f'img_list: {img_list}')
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label_list = []
        for phase in file_info['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
                # print(f'phase: {phase}, label: {self.dict[phase][0]}')
        data = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list] 
        # convert data to numpy array
        data = np.array(data)
        return data, label_list, file_info
    
    def read_feature(self, index):
        # load file info
        file_info = self.inputs_list[index]
        data = np.load(f'features\\{self.mode}\\{file_info["fileid"]}.npy', allow_pickle= True)
        return data['features'], data['label']
    
    def normalize(self, video, label, file_id= None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label
    
    def transform(self):
        if self.transform_mode:
            print("Apply training transform")
            return data_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                data_augmentation.RandomCrop(self.input_size),
                data_augmentation.RandomHorizontalFlip(0.5),
                data_augmentation.Resize(self.image_scale),
                data_augmentation.ToTensor(),
                data_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply validation transform")
            return data_augmentation.Compose([
                data_augmentation.CenterCrop(self.input_size),
                data_augmentation.Resize(self.image_scale),
                data_augmentation.ToTensor(),
            ])
            
    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes 
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            # print the shape of video before and after padded
            # print(f'video shape: {video[0].shape} -> {padded_video.shape}')
            return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1
    
prefix = os.getenv("DATA_PATH")
# load the numpy file
gloss_dict = np.load('Information_dict\\gloss_dict.npy', allow_pickle= True).item()


if __name__ == "__main__":
    dataset = VideoDataset(prefix= prefix, gloss_dict= gloss_dict, kernel_size= [('K', 3), ('P', 2)])
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )
    print(f'dataset length: {len(dataloader)}')
