import torch
import torch.nn as nn
import  torch.nn.functional as F

class BiGRU(nn.Module):
    def __init__(self, input_size, debug= False, hidden_siz= 512, num_layers= 1, dropout= 0.3, 
                 bidirectional= True, rnn_type= "GRU", num_classes= -1):
        super(BiGRU, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.debug = debug
        self.rnn = getattr(nn, self.rnn_type)(
            input_size= self.input_size,
            hidden_size= self.hidden_size,
            num_layers= self.num_layers,
            dropout= self.dropout,
            bidirectional= self.bidirectional,
        )
        
    def forward(self, src_feats, src_lens, hidden= None):
        