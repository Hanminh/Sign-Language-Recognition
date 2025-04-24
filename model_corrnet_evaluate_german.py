import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Modules import *
from Generate_Data.data_augmentation import *
import data_loader
import os
import torch
from Modules import BiLSTM
from Modules.BiLSTM import BiLSTM
from Modules.attention_corrnet import ResNet, BasicBlock
from model_corrnet_slr import SLR_Network
from torch.nn import CTCLoss
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from argument import *
import gc
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def encode_text(sample):
    encode_text = torch.tensor([])
    for i in range(len(sample[2])) :
        encode_text = torch.cat((encode_text, torch.tensor([sample[2][i], 0])))
        if i == len(sample[2]) - 1:
            encode_text = torch.cat((encode_text, torch.tensor([sample[2][i]])))
    return encode_text
  


# get the gloss_dict
prefix = os.getenv("DATA_PATH")
# prepare the gloss dictionary
gloss_dict = np.load('Information_dict/gloss_dict.npy', allow_pickle= True)
gloss_dict = gloss_dict.item()
id2gloss = []
id2gloss.append('<blank>')
for i in list(gloss_dict.keys()):
    id2gloss.append(gloss_dict[i])
# print(len(gloss_dict))

from pyctcdecode import build_ctcdecoder
dictionary = []
dictionary.append(' ')
for i in list(gloss_dict.keys()):
    dictionary.append(i + '|')

import jiwer
def calculate_wer(pred, true):
    pred_words = pred.split('|')[:-1]
    pred_str = ' '.join(pred_words)
    wer_score = jiwer.wer(true, pred_str)
    return wer_score


# Prepare the model
model = SLR_Network(num_classes= len(id2gloss) + 1, dictionary= dictionary)
model.to('cuda')
# criterion = CTCLoss(blank= 0, zero_infinity= True)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay= 0.0001)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= [40, 60], gamma= 1/5)
scaler = GradScaler()

# checkpoint = torch.load('/home/guest/Minh_20210605/Sign-Language-Recognition/Model/model_checkpoint_epoch_90.pth')
# model.load_state_dict(checkpoint['model_state_dict'], strict= False)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# print("load model 90e")

torch.cuda.empty_cache()


def evaluate_model(mode, batchsize= 1):
    
    # Prepare dataset
    dataset = data_loader.VideoDataset(prefix= prefix, gloss_dict= gloss_dict, kernel_size= [('K', 5), ('P', 2),('K', 5), ('P', 2)],
                                        mode= mode,transform_mode= False)
    dataset = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            collate_fn=dataset.collate_fn,
            pin_memory= True
        )

    running_loss = 0.0
    wer = 0.0
    model.eval()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataset)):
        
                input = sample[0].to('cuda', non_blocking=True)
                vid_len = sample[1]
                targets = sample[2]
                target_lengths = sample[3]
                # Forward pass
                with autocast():
                    output = model(input, vid_len)
                for i in range(batchsize):
                    wer += calculate_wer(output["predictions"][i], sample[-1][i])
                del  input, output, vid_len
                gc.collect()
                torch.cuda.empty_cache()

        epoch_loss = running_loss / len(dataset) / batchsize
        wer = wer / len(dataset) / batchsize
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{mode} loss: {epoch_loss}, {mode} wer: {wer}")
        
evaluate_model('train')
evaluate_model('test')
evaluate_model('dev')


