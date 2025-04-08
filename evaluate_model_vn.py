import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from Modules import *
from Generate_Data.data_augmentation import *
import data_loader_vn
import os
import torch
from Modules import BiLSTM
from Modules.BiLSTM import BiLSTM
from Modules.attention_corrnet import ResNet, BasicBlock
from slr_network import SLR_Network
from torch.nn import CTCLoss
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from argument import BATCHSIZE_EVAL, USE_GPU_EVAL, HIDDEN_SIZE, CONV_TYPE
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
import os
FEATURE_PATH = os.getenv("FEATURE_PATH")
INFORMATION_PATH = os.getenv("INFORMATION_PATH")
FEATURE_PATH = os.getenv("FEATURE_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

dataset = data_loader_vn.VideoDataset(id2gloss=vn_id2gloss, gloss2id=vn_gloss2id,
                                      kernel_size= [('K', 5), ('P', 4),('K', 5), ('P', 2)],
                                      mode= 'test', transform_mode= False, feature_folder= FEATURE_PATH,
                                      infor_folder= INFORMATION_PATH)

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size= BATCHSIZE_EVAL, 
    shuffle=False, 
    num_workers=0, 
    collate_fn=dataset.collate_fn,
    drop_last= True)

# Prepare the model
model = SLR_Network(num_classes= len(dictionary) + 1, dictionary= dictionary, conv_type= 3, hidden_size= 2048)
# model.to('cuda')
checkpoint = torch.load('DataDebug/model_checkpoint_epoch_100.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict= False)
if USE_GPU_EVAL:
    model = model.to('cuda')
torch.cuda.empty_cache()
model.eval()
wer = 0
with torch.no_grad():
    for i, sample in enumerate(dataloader):
        input_data = sample[0]
        if USE_GPU_EVAL:
            input_data = input_data.to('cuda')
        vid_len = sample[1]
        output = model(input_data, vid_len)
        for i in range(BATCHSIZE_EVAL):
            wer += calculate_wer(output['predictions'][i], sample[-1][i])
            print(f'Pred: {output["predictions"][i]}')
            print(f'True: {sample[-1][i]}')

print(f'WER: {wer / len(dataloader) / BATCHSIZE_EVAL}')
        
            



