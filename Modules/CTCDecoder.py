import torch.nn as nn
from pyctcdecode import build_ctcdecoder
import numpy as np

class CTCDecoder(object):
    def __init__(self, id2gloss, num_classes, beam_size, blank_id= 0):
        self.id2gloss = id2gloss
        self.num_classes = num_classes
        self.beam_size = beam_size
        self.blank_id = blank_id
        self.decoder = build_ctcdecoder(self.id2gloss)
    
    def decode_logits(self, logits):
        return self.decoder.decode(logits, self.beam_size)