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
      vid_len = vid_len = torch.tensor([sample[0].shape[1]]).to('cuda')
      target_lengths = sample[3].to('cuda')
      
      # input_model = sample[0]
      # vid_len = vid_len = torch.tensor([sample[0].shape[1]])
      # target_lengths = sample[3]
      
      # Forward pass
      output_model = model(input_model, vid_len)
      # input_lengths = torch.tensor([output_model["sequence_logits"].shape[0] for i in range(output_model['sequence_logits'].shape[1])], dtype=torch.long)      
      input_lengths = torch.tensor([output_model["sequence_logits"].shape[0] for i in range(output_model['sequence_logits'].shape[1])], dtype=torch.long).to('cuda')
      if output_model["predictions"] is not None:
          wer += calculate_wer(output_model["predictions"], sample[-1][0])
          num_wer = num_wer + 1
      optimizer.zero_grad()
      try:
          with autocast():
              loss = model.get_loss(output_model, input_lengths, sample[2], target_lengths)
          if loss is None:
              del loss, output_model, input_model, vid_len, input_lengths, target_lengths
              continue
          if np.isinf(loss.item()) or np.isnan(loss.item()) or loss is None:
              del loss, output_model, input_model, vid_len, input_lengths, target_lengths
              continue
          running_loss += loss.item()
          num_loss = num_loss + 1
          
          # Backward and optimize
          scaler.scale(loss).backward()
          scaler.step((optimizer))
          scaler.update()
      except Exception as e:
          print(sample[0])
          print(sample[0].shape)
          print(e)
          break
      # loss.detach()
      del loss, input_model, output_model, vid_len, sample, input_lengths, target_lengths
      del input_model
    #   gc.collect()
    #   torch.cuda.empty_cache()
      
  epoch_loss = running_loss / num_loss
  wer = wer / num_wer
  loss_histories.append(epoch_loss)
  wer_histories.append(wer)
  print(f'Epoch [{epoch+1}/{10}], Loss: {epoch_loss:.4f}, WER: {wer}, Num_wer: {num_wer}')
  if epoch == 4:
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
      }, f'/kaggle/working/model_checkpoint_epoch_{epoch}.pth')
  # torch.cuda.empty_cache()

# save the loss_histories
np.save('/kaggle/working/loss_histories.npy', loss_histories)

# save the model
torch.save(model.state_dict(), '/kaggle/working/model.pth')
np.save('/kaggle/working/wer_histories.npy', wer_histories)