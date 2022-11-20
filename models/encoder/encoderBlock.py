import torch.nn as nn
import copy
import subLayer.residualConnectionLayer as residualConnectionLayer

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, positional_ff, norm, dr_rate = 0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = residualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual2 = residualConnectionLayer(copy.deepcopy(norm, dr_rate))
        self.position_ff = positional_ff

    def forward(self, src, src_mask):
        out = src
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)
        return out