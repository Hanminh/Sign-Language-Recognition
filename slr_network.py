import torch.nn as nn
import torch.nn.functional as F
from Modules.BiLSTM import BiLSTM
from Modules.Convolution1D import Convolution1D
from Modules.attention_corrnet import BasicBlock, conv3x3, Get_Correlation, ResNet, pretrain_resnet18
from Modules.Loss import SeqKD
from Modules.CTCDecoder import CTCDecoder
from Modules.temporal_lifting_pool import TemporalConv
import numpy as np
import torch
import jiwer

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class SLR_Network(  nn.Module):
    def __init__(self, hidden_size= 1024, kernel_size=5,  num_classes= 1000, dictionary= None, T = 1., beam_size= 10):
        super(SLR_Network, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.T = T
        self.decoder = CTCDecoder(
            dictionary, 
            num_classes,
            beam_size= beam_size
        )
        
        self.BiLSTM = BiLSTM(
            input_size=self.hidden_size, 
            hidden_size= self.hidden_size // 2,
            num_classes= self.num_classes, 
            num_layers= 2,
            bidirectional= True)
        
        self.CorrNet = pretrain_resnet18()
        self.CorrNet.fc = Identity()
        # self.ConvNet = Convolution1D(
        #     input_size= self.num_classes, 
        #     hidden_size= self.hidden_size,
        #     num_classes= self.num_classes,
        #     kernel_size= self.kernel_size
        # )
        
        self.Temporal_Conv = TemporalConv(
            input_size= 512,
            hidden_size= self.hidden_size,
            num_classes= self.num_classes,
            conv_type= 2
        ) 
        
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.ctc_loss = nn.CTCLoss(blank= 0, zero_infinity= True)
        self.distillation_loss = SeqKD(T= self.T)
        
    # def calculate_wer(pred, true):
    #     pred_words = pred.split('|')[:-1]
    #     pred_str = ' '.join(pred_words)
    #     wer_score = jiwer.wer(true, pred_str)
    #     return wer_score
       
    def forward(self, feat, vid_len):
        batch, temp, channel, height, width = feat.shape
        feat = feat.permute(0, 2, 1, 3, 4) # Shape: (batch, channels, T, H, W)
        feat = self.CorrNet(feat)
        
        # Convolution1D
        feat = feat.view(batch, temp, -1).permute((0, 2, 1))
        out_conv = self.Temporal_Conv(feat, vid_len) 
        
        # BiLSTM 
        feat = out_conv["feature"].permute(2, 0, 1)
        out_lstm = self.BiLSTM(feat, [out_conv["feat_len"]])
        output = self.classifier(out_lstm["predictions"])
        # decode = self.decoder.decode_logits(output["sequence_logits"].squeeze().cpu().detach().numpy())
        logit_logprob = output.view(-1, self.num_classes).log_softmax(-1).squeeze().cpu().detach().numpy() 
        if np.isnan(logit_logprob).any() or np.isinf(logit_logprob).any():
            decode = None
        else:
            decode = self.decoder.decode_logits(logit_logprob)
        return {
            "feat_len": out_conv["feat_len"],
            "conv_logits": out_conv["conv_logits"],
            "loss_update_lift": out_conv["loss_LiftPool_u"],
            "loss_pred_lift": out_conv["loss_LiftPool_p"],
            "sequence_logits": output,
            "predictions": decode
        }
    
    def get_loss(self, output, input_len, label, label_len):
        loss = 0 
        # CTC Loss
        if torch.isnan(output["sequence_logits"]).any() or torch.isinf(output["sequence_logits"]).any():
            return None
        loss += self.ctc_loss(
            output["sequence_logits"].log_softmax(-1),
            label,
            input_len,
            label_len
        ).mean()
        
        # Distillation Loss
        if torch.isnan(output["conv_logits"]).any() or torch.isinf(output["conv_logits"]).any():
            return None
        loss += 25 * self.distillation_loss(
            output["conv_logits"].permute(2, 0, 1),
            output["sequence_logits"].detach()
        )
        
        
        loss += self.ctc_loss(
            output["conv_logits"].permute(2, 0, 1).log_softmax(-1),
            label,
            input_len,
            label_len
        ).mean()
        
        loss += 0.0005 * (output["loss_update_lift"] + output["loss_pred_lift"])
        
        return loss
        
        
        