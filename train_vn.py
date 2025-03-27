import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Modules import *
from Generate_Data.data_augmentation import *
import data_loader_vn
import os
import torch
from Modules import BiLSTM
from Modules.BiLSTM import BiLSTM
from Modules.attention_corrnet import ResNet, BasicBlock
from slr_network import SLR_Network
from torch.nn import CTCLoss
from argument import *
import gc
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import jiwer

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

dataset = data_loader_vn.VideoDataset()
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=True, 
    num_workers=0, 
    collate_fn=dataset.collate_fn)

model = SLR_Network(dictionary= dictionary, num_classes= len(dictionary) + 1 )
model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay= 0.0001)
loss_histories = []
wer_histories = []
torch.cuda.empty_cache()
scaler = GradScaler()

for epoch in range(5):
  running_loss = 0.0
  wer = 0
  model.train()
  num_wer = 1
  num_loss = 1
  for i, sample in tqdm(enumerate(dataloader)):
      input_model = sample[0].to('cuda')
      vid_len = vid_len = torch.tensor([sample[0].shape[1]])
      target_lengths = sample[3]
      target = sample[2]
      # forward pass
      with autocast():
        output = model(input_model, vid_len)
        input_lengths = torch.full(
            (output['sequence_logits'].shape[1],),
            output['sequence_logits'].shape[0],
            dtype=torch.long,
        )
        loss = model.get_loss(output, input_lengths, target, target_lengths)
        running_loss += loss.item()
        with torch.no_grad():
            for i in range(1):
                wer += calculate_wer(output['sequence_logits'][i], sample[-1][i])
        
        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
      loss.detach()
      del loss, input_model, output, vid_len
      gc.collect()
      torch.cuda.empty_cache()
      
  epoch_loss = running_loss / num_loss
  wer = wer / num_wer
  loss_histories.append(epoch_loss)
  wer_histories.append(wer)
  gc.collect()
  torch.cuda.empty_cache()
  print(f'Epoch [{epoch+1}/{10}], Loss: {epoch_loss:.4f}, WER: {wer}, Num_wer: {num_wer}')
  if epoch == 1:
      torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': epoch_loss,
      }, f'/kaggle/working/model_checkpoint_epoch_{epoch}.pth')

# save the loss_histories
np.save('/kaggle/working/loss_histories.npy', loss_histories)

# save the model
torch.save(model.state_dict(), '/kaggle/working/model.pth')
np.save('/kaggle/working/wer_histories.npy', wer_histories)