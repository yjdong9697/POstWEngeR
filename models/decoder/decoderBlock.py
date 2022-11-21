import torch.nn as nn
import copy
import subLayer.residualConnectionLayer as residualConnectionLayer

class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate = 0):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = residualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual2 = residualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual3 = residualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.cross_attention = cross_attention
        self.position_ff = position_ff

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residual1(out, lambda out : self.self_attention(query = out, key = out, value = out, mask = tgt_mask))
        out = self.residual2(out, lambda out : self.cross_attention(query = out, key = encoder_out, value = encoder_out, mask = src_tgt_mask))
        out = self.residual3(out, self.position_ff)
        return out