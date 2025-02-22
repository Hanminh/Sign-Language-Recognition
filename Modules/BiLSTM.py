import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):    
    def __init__(self, input_size, debug= False, hidden_size= 512, num_layers= 1, dropout= 0.3, 
                 bidirectional= True, rnn_type= "LSTM", num_classes= -1):
        super(BiLSTM, self).__init__()
        
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
        
        # src_feats: (max_src_len, batch_size, D)
        # src_lens: (batch_size)
        
        # Returns:
        # outputs: (max_src_len, batch_size, hidden_size * num_directions)
        # hidden: (num_layers, batch_size, hidden_size * num_directions)
        # print(f"src_feats: {src_feats.shape}")
        # print(f"src_lens: {src_lens}")
        
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            src_feats, src_lens[0]
        )
        # print(f"packed_emb: {packed_emb.data.shape}")
        
        if hidden is not None and self.rnn_type == "LSTM":
            half = int(hidden.size(0) / 2)
            hidden = (hidden[:half], hidden[half:])
            # print(hidden.shape)
        packed_outputs, hidden = self.rnn(packed_emb, hidden)
            
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outputs
        )
        
        if self.bidirectional:
            hidden = self.__cat_directions(hidden)
            
        if isinstance(hidden, tuple):
            hidden = hidden[0]
            
        return {
            "predictions": rnn_outputs,
            "hidden": hidden
        }
        
    def __cat_directions(self, hidden):
        
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            hidden = tuple([_cat(h) for h in hidden])
        else:
            hidden = _cat(hidden)
            
        return hidden
    
    