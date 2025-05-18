import torch.nn as nn
import torch.nn.functional as F
from Modules.BiLSTM import BiLSTM
from Modules.Classification_For_LSTM import BiLSTMClassifier
from Modules.BiLSTM import BiLSTM
from Modules.Convolution1D import TemporalConv
from Modules.attention_corrnet import BasicBlock, conv3x3, Get_Correlation, ResNet, pretrain_resnet18
from Modules.Loss_for_classifier import SeqKD
from Modules.CTCDecoder import CTCDecoder
from Modules.temporal_lifting_pool import TemporalConv as T1
from Modules.temporal_conv import TemporalConv as T2
import numpy as np
import torch
import jiwer

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class SLR_Network(  nn.Module):
    def __init__(self, hidden_size= 1024, kernel_size=5,  num_classes= 1000, dictionary= None, T = 1., beam_size= 50, conv_type =9, conv_improve= False, lstm_layers = 1, lstm_layers_classifier = 1, num_neighbors= [1, 3, 5] ):
        super(SLR_Network, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.T = T
        self.conv_type = conv_type
        self.conv_improve = conv_improve
        self.decoder = CTCDecoder(
            dictionary, 
            num_classes,
            beam_size= beam_size
        )
        
        # self.BiLSTM = BiLSTM(
        #     input_size=self.hidden_size, 
        #     hidden_size= self.hidden_size // 2,
        #     num_classes= self.num_classes, 
        #     num_layers= 2,
        #     bidirectional= True)
        
        self.BiLSTM_Classifier = BiLSTMClassifier(
            input_size=self.hidden_size,
            hidden_size= self.hidden_size // 2,
            num_layers= lstm_layers_classifier,
            bidirectional= True,
            num_classes= self.num_classes)
        
        self.BiLSTM = BiLSTM(
            input_size=self.hidden_size,
            hidden_size= self.hidden_size // 2,
            num_layers= lstm_layers,
            bidirectional= True,
            num_classes= self.num_classes)
        
        self.CorrNet = pretrain_resnet18(num_neighbors= num_neighbors)
        self.CorrNet.fc = Identity()
        # self.ConvNet = Convolution1D(
        #     input_size= self.num_classes, 
        #     hidden_size= self.hidden_size,
        #     num_classes= self.num_classes,
        #     kernel_size= self.kernel_size
        # )
        if self.conv_improve:
            self.Temporal_Conv = T1(
                input_size= 512,
                hidden_size= self.hidden_size,
                num_classes= self.num_classes,
                conv_type= self.conv_type
            )
        else:
            self.Temporal_Conv = T2(
                input_size= 512,
                hidden_size= self.hidden_size,
                num_classes= self.num_classes,
                conv_type= self.conv_type
            )
        
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # self.ctc_loss = torch.nn.CTCLoss(blank= 0, reduction= 'mean')
        self.distillation_loss = SeqKD(T= self.T)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
    # def calculate_wer(pred, true):
    #     pred_words = pred.split('|')[:-1]
    #     pred_str = ' '.join(pred_words)
    #     wer_score = jiwer.wer(true, pred_str)
    #     return wer_score
       
    def forward(self, feat, vid_len):
        if torch.isnan(feat).any() or torch.isinf(feat).any():
            print("Dữ liệu đầu vào có NaN hoặc Inf!")
        batch, temp, channel, height, width = feat.shape
        feat = feat.permute(0, 2, 1, 3, 4) # Shape: (batch, channels, T, H, W)
        feat = self.CorrNet(feat)
        
        # Convolution1D
        feat = feat.view(batch, temp, -1).permute((0, 2, 1))
        out_conv = self.Temporal_Conv(feat, vid_len) 
        
        # BiLSTM 
        feat = out_conv["feature"].permute(2, 0, 1)
        # print(f'Shape of outconv: {out_conv["feature"].shape}')
        out_lstm = self.BiLSTM(feat, [out_conv["feat_len"]])
        # print(f'Shape of outlstm: {out_lstm["predictions"].shape}')
        out_lstm_classifier = self.BiLSTM_Classifier(out_lstm["predictions"], [out_conv["feat_len"]])
        
        outconv_lstm_classifier = self.BiLSTM_Classifier(out_conv["feature"].permute(2, 0, 1), [out_conv["feat_len"]])
        
        # get the top-1 prediction
        predicted_class = out_lstm_classifier["logits"].argmax(-1)
        
        
        # decode = self.decoder.decode_logits(output["sequence_logits"].squeeze().cpu().detach().numpy())
        # logit_logprob = output.permute(1, 0, 2).log_softmax(-1).cpu().detach().numpy() 
        # if np.isnan(logit_logprob).any() or np.isinf(logit_logprob).any():
        #     decode = None
        # else:
        #     decode = self.decoder.decode_logits(logit_logprob)
        if self.conv_improve:
            return {
                "feat_len": out_conv["feat_len"],
                "loss_update_lift": out_conv["loss_LiftPool_u"],
                "loss_pred_lift": out_conv["loss_LiftPool_p"],
                # "predictions": decode,
                # "lstm_classifier": out_lstm["logits"],
                "lstm_classifier": out_lstm_classifier["logits"],
                "conv_lstm_classifier": outconv_lstm_classifier["logits"],
                "predicted_class": predicted_class,
            }
    
        else:
            return {
                "feat_len": out_conv["feat_len"],
                "lstm_classifier": out_lstm_classifier["logits"],
                "conv_lstm_classifier": outconv_lstm_classifier["logits"],
                "predicted_class": predicted_class
            }
            
    def get_loss(self, output, label, coef= [1, 1, 1]):
        total_loss = {}
        loss = 0 
        # CTC Loss
        # if torch.isnan(output["sequence_logits"]).any() or torch.isinf(output["sequence_logits"]).any():
        #     return None
        # # Distillation Loss
        # if torch.isnan(output["conv_logits"]).any() or torch.isinf(output["conv_logits"]).any():
        #     return None
        # assert (input_len >= label_len).all()
        # if (input_len <= 0).any() or (label_len <= 0).any():
        #     print("input_lengths hoặc target_lengths có giá trị <= 0!")
        #     print(label)
        #     print(label_len)
        #     print(input_len)
        # if torch.isnan(input_len).any() or torch.isnan(label_len).any():
        #     print("NaN xuất hiện trong input_lengths hoặc target_lengths!")
        #     print(label)
        #     print(label_len)
        #     print(input_len)

        # if torch.isnan(output['conv_logits']).any() or torch.isnan(output["sequence_logits"]).any():
        #     print("NaN detected in prediction_logits before log_softmax!")
        #     print(label)
        #     torch.save(output, '/home/guest/Minh_20210605/Model_VN/debug_data_1.pt')


        # if torch.isinf(output['conv_logits']).any() or torch.isinf(output["sequence_logits"]).any():
        #     print("Inf detected in prediction_logits before log_softmax!")
        #     print(label)
        #     torch.save(output, '/home/guest/Minh_20210605/Model_VN/debug_data_1.pt')

        # total_loss['Seq'] = self.ctc_loss(
        #     output["sequence_logits"].log_softmax(-1),
        #     label,
        #     input_len,
        #     label_len
        # ).mean()
        
        # total_loss['Conv'] = self.ctc_loss(
        #     output["conv_logits"].permute(2, 0, 1).log_softmax(-1),
        #     label,
        #     input_len,
        #     label_len
        # ).mean()
        
        total_loss['Dist'] = self.distillation_loss(
            output["conv_lstm_classifier"],
            output["lstm_classifier"].detach()
        )
        
        total_loss['CrossEntropy'] = self.cross_entropy_loss(
            output["lstm_classifier"],
            label
        )
        
        total_loss['Conv_CrossEntropy'] = self.cross_entropy_loss(
            output["conv_lstm_classifier"],
            label
        )

        # loss += total_loss['Seq']
        # loss += total_loss['Conv']
        loss += coef[0] * total_loss['CrossEntropy']
        loss += coef[1] *total_loss['Conv_CrossEntropy']
        loss += coef[2] * total_loss['Dist']
        if self.conv_improve:
            loss += 0.0005 * (output["loss_update_lift"] + output["loss_pred_lift"])
        
        return loss
        