import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolution1D(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type= 2, batch_norm= False, num_classes= -1, kernel_size= 5):
        super(Convolution1D, self).__init__()
        self.batch_norm = batch_norm
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.kernel_size = kernel_size
        self.norm = False
        self.conv1d = nn.Conv1d(
            in_channels= self.input_size,
            out_channels= self.hidden_size,
            kernel_size= self.kernel_size,
            padding= 0,
            stride= 1
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels= self.hidden_size,
            out_channels= self.hidden_size,
            kernel_size= self.kernel_size,
            padding= 0,
            stride= 1
        )
        self.pool = nn.MaxPool1d(kernel_size= 2, ceil_mode= False)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(self.hidden_size)
        self.relu = nn.ReLU(inplace= True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes) 
        
    def forward(self, feature, feat_len):
        
        # forward pass
        x = self.conv1d(feature)
        x = self.pool(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.pool(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        
        new_feat_len = self.compute_feat_len(feat_len)
        logits = self.fc(x.transpose(1,2)).transpose(1,2)
        
        return {
            "feature": x,
            "feat_len": new_feat_len,
            "logits": logits
        }
        # update feature_len after convolution and pooling
    def compute_feat_len(self, feat_len):
        # First Conv1d (kernel_size=5, padding=0, stride=1)
        feat_len = feat_len - self.kernel_size + 1  
        # First MaxPool1d (kernel_size=2, stride=2)
        feat_len = feat_len // 2
        # Second Conv1d (same params)
        feat_len = feat_len - self.kernel_size + 1
        # Second MaxPool1d (same params)
        feat_len = feat_len // 2

        return feat_len