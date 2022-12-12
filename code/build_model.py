import torch.nn as nn
import copy
import torch
from models.embedding.tokenEmbedding import TokenEmbedding
from models.embedding.positionalEncoding import PositionalEncoding
from models.embedding.transformerEmbedding import TransformerEmbedding
from models.subLayer.multiHeadAttentionLayer import MultiHeadAttentionLayer
from models.subLayer.positionWiseFeedForwardLayer import PositionWiseFeedForwardLayer
from models.encoder.encoder import Encoder
from models.encoder.encoderBlock import EncoderBlock
from models.decoder.decoder import Decoder
from models.decoder.decoderBlock import DecoderBlock
from models.transformer.transformer import Transformer

def build_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device("cpu"),
                max_len = 256,
                d_embed = 512,
                n_layer = 6,
                d_model = 512,
                h = 8,
                d_ff = 2048,
                dr_rate = 0.1,
                norm_eps = 1e-5):

    src_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = src_vocab_size)
    tgt_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = tgt_vocab_size)
    pos_embed = PositionalEncoding(
                                   d_embed = d_embed,
                                   max_len = max_len,
                                   device = device)

    src_embed = TransformerEmbedding(
                                     token_embed = src_token_embed,
                                     pos_embed = copy.copy(pos_embed),
                                     dr_rate = dr_rate)

    tgt_embed = TransformerEmbedding(
                                     token_embed = tgt_token_embed,
                                     pos_embed = copy.copy(pos_embed),
                                     dr_rate = dr_rate)

    attention = MultiHeadAttentionLayer(
                                        d_embed = d_embed,
                                        d_model = d_model,
                                        h = h,
                                        dr_rate = dr_rate)
    position_ff = PositionWiseFeedForwardLayer(
                                               fc1 = nn.Linear(d_embed, d_ff),
                                               fc2 = nn.Linear(d_ff, d_embed),
                                               dr_rate = dr_rate)
    norm = nn.LayerNorm(d_embed, eps = norm_eps)

    encoder_block = EncoderBlock(
                                 self_attention = copy.copy(attention),
                                 position_ff = copy.copy(position_ff),
                                 norm = copy.copy(norm),
                                 dr_rate = dr_rate)
    decoder_block = DecoderBlock(
                                 self_attention = copy.copy(attention),
                                 cross_attention = copy.copy(attention),
                                 position_ff = copy.copy(position_ff),
                                 norm = copy.copy(norm),
                                 dr_rate = dr_rate)

    encoder = Encoder(
                      encoder_block = encoder_block,
                      n_layer = n_layer,
                      norm = copy.copy(norm))
    decoder = Decoder(
                      decoder_block = decoder_block,
                      n_layer = n_layer,
                      norm = copy.copy(norm))
    generator = nn.Sequential(
                      nn.Linear(d_model, 1024),
                      nn.ReLU(),
                      nn.Linear(1024, 1)
    )

    # generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
                        src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device

    return model