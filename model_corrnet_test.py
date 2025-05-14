import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data_loader_vn_wacv
import torch
from Modules import BiLSTM
from Modules.BiLSTM import BiLSTM
from model_corrnet_slr import SLR_Network
from argument import BATCHSIZE_EVAL, HIDDEN_SIZE_CORRNET, CONV_TYPE_CORRNET_CNN_IMPROVE
from dotenv import load_dotenv
load_dotenv()
import os
from tqdm import tqdm
import jiwer
import cv2 as cv
from Generate_Data import data_augmentation
from Generate_Data.data_augmentation import *

IMG_SIZE = (256, 256)
VIDEO_PATH = '00693_bạn gái.mp4'
INFORMATION_PATH = os.getenv("INFORMATION_PATH")
vn_dictionary = np.load(f'{INFORMATION_PATH}/vn_dictionary.npy', allow_pickle=True).item()
dictionary = []
dictionary.append(' ')
for i in list(vn_dictionary.keys()):
    dictionary.append(i + '|')
    
# load the model
def load_model(path= 'DataDebug/model_checkpoint_epoch_100.pth'):
    model = SLR_Network(num_classes= len(dictionary) + 1, dictionary= dictionary, conv_type= CONV_TYPE_CORRNET_CNN_IMPROVE, hidden_size= HIDDEN_SIZE_CORRNET)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict= False)
    return model

# load the video
def load_video(path= VIDEO_PATH):
    capture = cv.VideoCapture(path)
    frames = []
    num = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = frame[:, 220:1060]
        frame = cv.resize(frame, (256, 256), cv.INTER_LANCZOS4)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if num % 5 == 0:
            frames.append(frame)
        num += 1
    capture.release()
    return np.array(frames)

# prepare data for model
def prepare_data(video, kernel_sizes= [('K', 5), ('P', 4),('K', 5), ('P', 2)]):
    data_aug = data_augmentation.Compose([
        data_augmentation.CenterCrop((224, 224)),
        data_augmentation.Resize(1),
        data_augmentation.ToTensor()
    ])
    video, label = data_aug(video, '')
    video = video.float() / 127.5 - 1
    video = video.unsqueeze(0)
    left_pad = 0
    last_stride = 1
    total_stride = 1
    # kernel_sizes = ['K5', "P4", 'K5', "P2"]
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
    
    return padded_video

def prediction(model_path, video_path):
    model = load_model(model_path)
    video = load_video(video_path)
    video = prepare_data(video)
    model.eval()
    len_vid = torch.Tensor([video.shape[1]])
    # video = video.to('cuda')
    # model.to('cuda')
    with torch.no_grad():
        output = model(video, len_vid)
        pred = output["predictions"]
    return pred

predicted_sent = prediction('DataDebug/model_checkpoint_epoch_100.pth', VIDEO_PATH)
print(predicted_sent)

    
