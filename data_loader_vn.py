from Generate_Data import data_augmentation
from Generate_Data.data_augmentation import *
import os
import cv2
import torch
import numpy as np
import pyarrow as pa
import pandas as pd
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler
import ast
from dotenv import load_dotenv
load_dotenv()
import os
FEATURE_PATH = os.getenv("FEATURE_PATH")
INFORMATION_PATH = os.getenv("INFORMATION_PATH")
FEATURE_PATH = os.getenv("FEATURE_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")

class VideoDataset(data.Dataset):
    def __init__(self,  drop_ratio=1, image_scale= 1.0, kernel_size= [('K', 5), ('P', 2),('K', 5), ('P', 2)], input_size= 224, mode= 'train', data_type= 'video', feature_folder= None, infor_folder= None, transform_mode= False, id2gloss= None, gloss2id= None):
        self.mode= mode
        self.transform_mode= transform_mode
        self.image_scale= image_scale
        global kernel_sizes
        kernel_sizes= kernel_size
        self.data_type= data_type
        self.feature_folder = feature_folder
        self.input_size= input_size
        self.drop_ratio= drop_ratio
        self.infor_folder= infor_folder
        # load the pickle file
        #using for Kaggle notebook
        if self.infor_folder:
            self.inputs_list = pd.read_csv(f'{infor_folder}/vn_data_{mode}.csv')
        else :
            self.inputs_list= pd.read_csv(f'{INFORMATION_PATH}/vn_data_{mode}.csv') 
        self.inputs_list['keys'] = self.inputs_list['keys'].apply(ast.literal_eval)
        self.inputs_list['id'] = self.inputs_list['id'].apply(ast.literal_eval)
        self.id2gloss= id2gloss
        self.gloss2id= gloss2id
        self.data_aug = self.transform()
        
    def __getitem__(self, index):
        if self.data_type == 'video':
            input_data, label, sent, vid_len = self.read_video(index)
            input_data, label = self.normalize(input_data, label, sent)
            # return {
            #     'video': input_data,
            #     'label': torch.LongTensor(label),
            #     'info': sent,
            #     'vid_len': vid_len
            # }
            return input_data, torch.LongTensor(label), sent, vid_len
        # elif self.data_type == 'feature':
        #     input_data, label = self.read_feature(index)
        #     return input_data, torch.tensor([label]), self.inputs_list[index]['label'], vid_len
    
    def read_video(self, index):
        # load file info
        sent, keys, ids = self.inputs_list.loc[index]
        if self.feature_folder:
            img_folders = [self.feature_folder + f'/{id}' for id in ids]
        else :
            img_folders = [FEATURE_PATH + f'\\{id}' for id in ids]
        # print(img_folder)
        # print(sent, keys, ids)
        # print(img_folders)
        img_list = []
        for folder in img_folders:
                # print(folder)
                img_paths = os.listdir(folder)
                # sort the image paths by name
                img_paths.sort()
                # print(img_paths)
                for path in img_paths:
                    if self.feature_folder:
                        img_list.append(f'{folder}/{path}')
                    else:
                        img_list.append(f'{folder}\\{path}')

        # label_list = [int(id) for id in ids]
        label_list = []
        # convert from id of folder to id of gloss 
        for id in ids:
            word = self.id2gloss[id]
            label_list.append(self.gloss2id[word])
        
        data = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list] 
        # convert data to numpy array
        data = np.array(data)
        return data, label_list, sent, data.shape[1]
    
    # def read_feature(self, index):
    #     # load file info
    #     file_info = self.inputs_list[index]
    #     data = np.load(f'features\\{self.mode}\\{file_info["fileid"]}.npy', allow_pickle= True)
    #     return data['features'], data['label']
    
    def normalize(self, video, label, file_id= None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label
    
    def transform(self):
        if self.transform_mode:
            print("Apply training transform")
            return data_augmentation.Compose([
                # data_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                data_augmentation.RandomCrop(self.input_size),
                data_augmentation.RandomHorizontalFlip(0.5),
                data_augmentation.Resize(self.image_scale),
                data_augmentation.ToTensor()
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
        video, label, info, vid_len = list(zip(*batch))
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes 
        # print(kernel_sizes)
        kernel_sizes
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
    
# prefix = os.getenv("DATA_PATH")
# # load the numpy file
# gloss_dict = np.load('Information_dict\\gloss_dict.npy', allow_pickle= True).item()


# if __name__ == "__main__":
#     dataset = VideoDataset(prefix= prefix, gloss_dict= gloss_dict, kernel_size= [('K', 3), ('P', 2)])
#     dataloader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=1,
#         shuffle=True,
#         drop_last=True,
#         num_workers=0,
#         collate_fn=dataset.collate_fn
#     )
#     print(f'dataset length: {len(dataloader)}')
