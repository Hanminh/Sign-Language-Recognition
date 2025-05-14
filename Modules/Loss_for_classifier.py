import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits):
        prediction_logits = F.log_softmax(prediction_logits / self.T, dim=-1)
        ref_logits = F.softmax(ref_logits / self.T, dim=-1)
        loss = self.kdloss(prediction_logits, ref_logits) * self.T * self.T
        return loss

