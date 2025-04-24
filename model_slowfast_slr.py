import pdb
import copy
from Modules.CTCDecoder import CTCDecoder
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Modules.Loss import SeqKD
from Modules.BiLSTM import BiLSTM
from Modules.temporal_conv import TemporalConv
import slowfast_modules.slowfast as slowfast
import importlib

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, load_pkl, slowfast_config, slowfast_args=[],
            use_bn=False, hidden_size=1024, dictionary=None, loss_weights=None,
            weight_norm=True, share_classifier=True, beam_size= 10, T= 1.0
    ):
        super(SLRModel, self).__init__()
        self.T = T
        self.decoder = None
        self.CTCLoss = torch.nn.CTCLoss(reduction='mean', blank= 0)
        self.dist_loss = SeqKD(T=self.T)
        self.hidden_size = hidden_size
        # self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(slowfast, c2d_type)(slowfast_config=slowfast_config, slowfast_args=slowfast_args,
                                                  load_pkl=load_pkl)

        self.conv1d = TemporalConv(input_size=2304,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        
        self.decoder = CTCDecoder(
            dictionary,
            num_classes,
            beam_size=beam_size
        )
        
        self.temporal_model = BiLSTM(rnn_type='LSTM',
                                     input_size=self.hidden_size, hidden_size=self.hidden_size // 2,
                                     num_layers=2,
                                     bidirectional=True)
        
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            framewise = self.conv2d(x.permute(0,2,1,3,4))
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        lgt = conv1d_outputs['feat_len']
        # print(conv1d_outputs['visual_feat'].shape)
        # print(lgt)
        tm_outputs = self.temporal_model(conv1d_outputs['visual_feat'], [lgt])
        outputs = self.classifier(tm_outputs['predictions'])
        # pred = None if self.training \
        #     else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        # conv_pred = None if self.training \
        #     else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        # pred = self.decoder.decode_logits(outputs)
        logits_logprob = outputs.permute(1, 0, 2).log_softmax(-1).cpu().detach().numpy()
        pred = self.decoder.decode_logits(logits_logprob)
        return {
            #"framewise_features": framewise,
            #"visual_features": conv1d_outputs['visual_feat'],
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            # "conv_sents": conv_pred,
            "predictions": pred,
        }

    def get_loss(self, output, input_len, label, label_len):
        loss = 0
        loss += self.CTCLoss(
            output['sequence_logits'].log_softmax(-1),
            label,
            input_len,
            label_len
        ).mean()
        
        loss += self.CTCLoss(
            output['conv_logits'].log_softmax(-1),
            label,
            input_len,
            label_len
        )
        
        loss += 25 * self.dist_loss(
            output['conv_logits'],
            output['sequence_logits']
        )
        
        return loss

    # def criterion_init(self):
    #     self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
    #     self.loss['distillation'] = SeqKD(T=8)
    #     return self.loss
