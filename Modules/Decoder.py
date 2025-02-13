import os
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id

        # Build vocabulary from the gloss dictionary
        vocab = [self.i2g_dict.get(i, "") for i in range(num_classes)]

        # Initialize pyctcdecode decoder
        self.ctc_decoder = build_ctcdecoder(
            labels=vocab
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        BeamSearch decoding using pyctcdecode
        Shape of nn_output:
            - Input: nn_output (B, T, N), should be passed through softmax layer if `probs` is False
            - Output: Decoded sequences
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu().numpy()

        ret_list = []
        for batch_idx, length in enumerate(vid_lgt):
            # Take the valid portion of the sequence based on `vid_lgt`
            logits = nn_output[batch_idx, :length]
            beam_result = self.ctc_decoder.decode(logits)

            # Map decoded indices back to gloss labels
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(beam_result)])

        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        '''
        Maximum likelihood decoding
        Shape of nn_output:
            - Input: nn_output (B, T, N)
            - Output: Decoded sequences
        '''
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, _ = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]

            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered

            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
