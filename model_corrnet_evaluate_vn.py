import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from Modules import *
from Generate_Data.data_augmentation import *
import data_loader_vn_wacv
import os
import torch
from Modules import BiLSTM
from Modules.BiLSTM import BiLSTM
from Modules.attention_corrnet import ResNet, BasicBlock
from model_corrnet_slr import SLR_Network
from torch.nn import CTCLoss
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from argument import BATCHSIZE_TRAIN, HIDDEN_SIZE_CORRNET, GAMMA, EPOCH, CONV_TYPE_CORRNET_CNN_IMPROVE, REGULARIZATION
import gc
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

INFORMATION_PATH = os.getenv("INFORMATION_PATH")
FEATURE_PATH = os.getenv("FEATURE_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")

def encode_text(sample):
    encode_text = torch.tensor([])
    for i in range(len(sample[2])) :
        encode_text = torch.cat((encode_text, torch.tensor([sample[2][i], 0])))
        if i == len(sample[2]) - 1:
            encode_text = torch.cat((encode_text, torch.tensor([sample[2][i]])))
    return encode_text

import jiwer
def calculate_wer(pred, true):
    pred_words = pred.split('|')[:-1]
    pred_str = ' '.join(pred_words)
    wer_score = jiwer.wer(true, pred_str)
    return wer_score


vn_id2gloss = np.load(f'{INFORMATION_PATH}/vn_id2gloss.npy', allow_pickle=True).item()

vn_dictionary = np.load(f'{INFORMATION_PATH}/vn_dictionary.npy', allow_pickle=True).item()
dictionary = []
dictionary.append(' ')
for i in list(vn_dictionary.keys()):
    dictionary.append(i + '|')

vn_gloss2id = np.load(f'{INFORMATION_PATH}/vn_gloss2id.npy', allow_pickle=True).item()

dataset = data_loader_vn_wacv.VideoDataset(id2gloss=vn_id2gloss, gloss2id=vn_gloss2id,
                                      kernel_size= [('K', 5), ('P', 4),('K', 5), ('P', 2)], mode= 'speaker_all', transform_mode= True, feature_folder= FEATURE_PATH,
                                      infor_folder= INFORMATION_PATH)

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size= BATCHSIZE_TRAIN, 
    shuffle=False, 
    num_workers=0, 
    collate_fn=dataset.collate_fn,
    drop_last= True)

# Prepare the model
model = SLR_Network(num_classes= len(dictionary) + 1, dictionary= dictionary, conv_type= CONV_TYPE_CORRNET_CNN_IMPROVE, hidden_size= HIDDEN_SIZE_CORRNET)
model.to('cuda')
wer_test = 0 
model.eval()
with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader)):
            input = sample[0].to('cuda', non_blocking=True)
            # vid_len = sample[1].to('cuda', non_blocking=True)
            # targets = sample[2].to('cuda', non_blocking=True)
            # target_lengths = sample[3].to('cuda', non_blocking=True)
            vid_len = sample[1]
            targets = sample[2]
            target_lengths = sample[3]

            output = model(input, vid_len)
            target_lengths = sample[3]
            for i in range(BATCHSIZE_TRAIN):
                wer_test += calculate_wer(output["predictions"][i], sample[-1][i])
            
            del input, output, vid_len

print(f'WER: {wer_test / len(dataloader)}')