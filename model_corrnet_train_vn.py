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
from model_corrnet_slr import SLR_Network
from torch.nn import CTCLoss
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from argument import BATCHSIZE_TRAIN, HIDDEN_SIZE_CORRNET, GAMMA, EPOCH, CONV_TYPE_CORRNET, REGULARIZATION
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

dataset_train = data_loader_vn.VideoDataset(id2gloss=vn_id2gloss, gloss2id=vn_gloss2id,
                                      kernel_size= [('K', 5), ('P', 4),('K', 5), ('P', 2)], mode= 'train', transform_mode= True, feature_folder= FEATURE_PATH,
                                      infor_folder= INFORMATION_PATH)

dataloader_train = torch.utils.data.DataLoader(
    dataset_train, 
    batch_size= BATCHSIZE_TRAIN, 
    shuffle=False, 
    num_workers=0, 
    collate_fn=dataset_train.collate_fn,
    drop_last= True)

dataset_dev = data_loader_vn.VideoDataset(id2gloss=vn_id2gloss, gloss2id=vn_gloss2id,
                                      kernel_size= [('K', 5), ('P', 4),('K', 5), ('P', 2)], mode= 'dev', transform_mode= False, feature_folder= FEATURE_PATH,
                                      infor_folder= INFORMATION_PATH)

dataloader_dev = torch.utils.data.DataLoader(
    dataset_dev, 
    batch_size=1, 
    shuffle=True, 
    num_workers=0, 
    collate_fn=dataset_dev.collate_fn,
    drop_last= True)


# Prepare the model
model = SLR_Network(num_classes= len(dictionary) + 1, dictionary= dictionary, conv_type= CONV_TYPE_CORRNET, hidden_size= HIDDEN_SIZE_CORRNET)
model.to('cuda')
# criterion = CTCLoss(blank= 0, zero_infinity= True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay= REGULARIZATION) 
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= [40, 60], gamma= GAMMA)
scaler = GradScaler()

loss_histories_train = []
wer_histories_train = []
loss_histories_dev = []
wer_histories_dev = []
torch.cuda.empty_cache()

model.train()
for epoch in range(0, EPOCH):
  
    running_loss_train = 0.0
    wer_train = 0.0
    running_loss_dev = 0.0
    wer_dev = 0.0
    model.train()
    for i, sample in tqdm(enumerate(dataloader_train)):
        
        input = sample[0].to('cuda', non_blocking=True)
        # vid_len = sample[1].to('cuda', non_blocking=True)
        # targets = sample[2].to('cuda', non_blocking=True)
        # target_lengths = sample[3].to('cuda', non_blocking=True)
        vid_len = sample[1]
        targets = sample[2]
        target_lengths = sample[3]

        # Forward pass
        with autocast():
            output = model(input, vid_len)
            input_lengths = torch.full(
                (output['sequence_logits'].shape[1],), 
                output['sequence_logits'].shape[0], 
                dtype=torch.long
            )
            target_lengths = sample[3]
            loss = model.get_loss(output, input_lengths, sample[2], target_lengths)
        
        # loss = model.get_loss(output, input_lengths, sample[2], target_lengths)
        running_loss_train += loss.item()
        with torch.no_grad():
            for i in range(1):
                wer_train += calculate_wer(output["predictions"][i], sample[-1][i])
        
        # Backward and optimize
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update learning rate 
        scheduler.step()

        loss.detach()
        del loss, input, output, vid_len
        gc.collect()
        torch.cuda.empty_cache()
    epoch_loss = running_loss_train / len(dataloader_train)
    wer_train = wer_train / len(dataloader_train)
    loss_histories_train.append(epoch_loss)
    wer_histories_train.append(wer_train)
    
    # dev 
    model.eval()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader_dev)):
            input = sample[0].to('cuda', non_blocking=True)
            # vid_len = sample[1].to('cuda', non_blocking=True)
            # targets = sample[2].to('cuda', non_blocking=True)
            # target_lengths = sample[3].to('cuda', non_blocking=True)
            vid_len = sample[1]
            targets = sample[2]
            target_lengths = sample[3]

            output = model(input, vid_len)
            input_lengths = torch.full(
                (output['sequence_logits'].shape[1],), 
                output['sequence_logits'].shape[0], 
                dtype=torch.long
            )
            target_lengths = sample[3]
            loss = model.get_loss(output, input_lengths, sample[2], target_lengths)

            running_loss_dev += loss.item()
            for i in range(1):
                wer_dev += calculate_wer(output["predictions"][i], sample[-1][i])
            
            del loss, input, output, vid_len
    
    running_loss_dev = running_loss_dev / len(dataloader_dev)
    wer_dev = wer_dev / len(dataloader_dev)
    loss_histories_dev.append(running_loss_dev)
    wer_histories_dev.append(wer_dev)
    
    print(f'Epoch [{epoch+1}/{80}], Loss: {epoch_loss:.4f}, Wer: {wer_train:.4f}, Dev Loss: {running_loss_dev:.4f}, Dev Wer: {wer_dev:.4f}')
    
    gc.collect()
    torch.cuda.empty_cache()
    if (epoch) % 10 == 0 :
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        }, f'{MODEL_SAVE_PATH}/model_checkpoint_epoch_{epoch}.pth')
        

torch.save({
  'epoch': epoch,
  'model_state_dict': model.state_dict(),
  'optimizer_state_dict': optimizer.state_dict(),
  'loss': epoch_loss,
}, f'{MODEL_SAVE_PATH}/final_model.pth')
# save the loss_histories
np.save(f'{MODEL_SAVE_PATH}/loss_histories.npy', loss_histories_train)
np.save(f'{MODEL_SAVE_PATH}/wer_histories.npy', wer_histories_train)
np.save(f'{MODEL_SAVE_PATH}/loss_histories_dev.npy', loss_histories_dev)
np.save(f'{MODEL_SAVE_PATH}/wer_histories_dev.npy', wer_histories_dev)
# save the model
torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/Model_VN/model.pth')