import torch.nn as nn
import torch.nn.functional as F
from Modules.BiLSTM import BiLSTM
from Modules.Convolution1D import Convolution1D
from Modules.correlationNet import BasicBlock, conv3x3, Get_Correlation, ResNet
from Modules.Loss import SeqKD
class SLR_Network(nn.Module):
    def __init__(self, hidden_size= 1024, kernel_size=5,  num_classes= 1024):
        super(SLR_Network, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        
        self.BiLSTM = BiLSTM(
            input_size=self.hidden_size, 
            hidden_size= self.hidden_size // 2, 
            num_classes= self.num_classes, 
            bidirectional= True)
        
        self.CorrNet = ResNet(
            block= BasicBlock,
            layers= [2, 2, 2, 2],
            num_classes= self.num_classes)
        
        self.ConvNet = Convolution1D(
            input_size= self.num_classes, 
            hidden_size= self.hidden_size,
            num_classes= self.num_classes,
            kernel_size= self.kernel_size
        )
        
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # self.ctc_loss = nn.CTCLoss(blank= 0, zero_infinity= True)
        # self.distillation_loss = SeqKD(T= 8)
        
    def forward(self, feat, vid_len):
        batch, temp, channel, height, width = feat.shape
        feat = feat.permute(0, 2, 1, 3, 4) # Shape: (batch, channels, T, H, W)
        feat = self.CorrNet(feat)
        
        # Convolution1D
        feat = feat.view(batch, temp, -1).permute((0, 2, 1))
        out_conv = self.ConvNet(feat, vid_len) 
        
        # BiLSTM 
        feat = out_conv["feature"].permute(2, 0, 1)
        out_lstm = self.BiLSTM(feat, [out_conv["feat_len"]])
        output = self.classifier(out_lstm["predictions"])
        return {
            "feat_len": out_conv["feat_len"],
            "conv_logits": out_conv["logits"],
            "lstm_predictions": out_lstm["predictions"],
            "lstm_hidden": out_lstm["hidden"],
            "sequence_logits": output
        }
    
    def get_loss(self, output, input_len, label, label_len):
        loss = 0 
        
        # CTC Loss
        loss += self.ctc_loss(
            output["sequence_logits"].log_softmax(-1),
            label,
            input_len,
            label_len
        ).mean()
        
        # # Distillation Loss
        # loss += 25 * self.distillation_loss(
        #     output["conv_logits"],
        #     output["sequence_logits"].detach()
        # )
        
        # loss += self.ctc_loss(
        #     output["conv_logits"].permute(2, 0, 1).log_softmax(-1),
        #     label,
        #     input_len,
        #     label_len
        # )
        
        return loss
        
        
        