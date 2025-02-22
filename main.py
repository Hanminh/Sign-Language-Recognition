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
from Modules.correlationNet import ResNet, BasicBlock
from slr_network import SLR_Network
from torch.nn import CTCLoss
from argument import *
import gc
from tqdm import tqdm
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
gloss_dict = np.load('Information_dict\\gloss_dict.npy', allow_pickle= True)
gloss_dict = gloss_dict.item()
id2gloss = []
id2gloss.append('<blank>')
for i in list(gloss_dict.keys()):
    id2gloss.append(gloss_dict[i])
# print(len(gloss_dict))


# Prepare dataset
dataset = data_loader.VideoDataset(prefix= prefix, gloss_dict= gloss_dict, kernel_size= [('K', 3), ('P', 2)])
dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

# Prepare the model
model = SLR_Network(dict_size= len(id2gloss) + 1)
model.to('cuda')
criterion = CTCLoss(blank= 0, zero_infinity= True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_histories = []
torch.cuda.empty_cache()
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    running_loss = 0.0
    for i, sample in enumerate(dataloader):
        input = sample[0]
        vid_len = sample[1]
        input = input.to('cuda')
        # encode_seq = encode_text(sample)
        
        # Forward pass
        output = model(input, vid_len)
        input_lengths = torch.tensor([output["sequence_logits"].shape[0] for i in range(output['sequence_logits'].shape[1])], dtype=torch.long)
        target_lengths = sample[3]

        loss = criterion(
            output["sequence_logits"].log_softmax(-1),  # Shape (T, N, C)
            sample[2],  # sequence
            input_lengths,  #  (batch_size,)
            target_lengths,  # (batch_size,)
        )
        running_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del loss
        del input
        # gc.collect()
    epoch_loss = running_loss / len(dataloader)
    loss_histories.append(epoch_loss)
    
# save the loss_histories
np.save('loss_histories.npy', loss_histories)
    
# save the model
torch.save(model.state_dict(), 'model.pth')

        
                