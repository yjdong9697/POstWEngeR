import torch.nn as nn
import copy
import torch
import embedding.tokenEmbedding as tokenEmbedding
import embedding.positionalEncoding as positionalEncoding
import embedding.transformerEmbedding as transformerEmbedding
import subLayer.multiHeadAttentionLayer as multiHeadAttentionLayer
import subLayer.positionWiseFeedForwardLayer as positionWiseFeedForwardLayer
import encoder.encoder as Encoder
import encoder.encoderBlock as EncoderBlock
import decoder.decoder as Decoder
import decoder.decoderBlock as decoderBlock
import transformer.transformer as transformer

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

    src_token_embed = tokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = src_vocab_size)
    tgt_token_embed = tokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = tgt_vocab_size)
    pos_embed = positionalEncoding(
                                   d_embed = d_embed,
                                   max_len = max_len,
                                   device = device)

    src_embed = transformerEmbedding(
                                     token_embed = src_token_embed,
                                     pos_embed = copy(pos_embed),
                                     dr_rate = dr_rate)
    tgt_embed = transformerEmbedding(
                                     token_embed = tgt_token_embed,
                                     pos_embed = copy(pos_embed),
                                     dr_rate = dr_rate)

    attention = multiHeadAttentionLayer(
                                        d_model = d_model,
                                        h = h,
                                        qkv_fc = nn.Linear(d_embed, d_model),
                                        out_fc = nn.Linear(d_model, d_embed),
                                        dr_rate = dr_rate)
    position_ff = positionWiseFeedForwardLayer(
                                               fc1 = nn.Linear(d_embed, d_ff),
                                               fc2 = nn.Linear(d_ff, d_embed),
                                               dr_rate = dr_rate)
    norm = nn.LayerNorm(d_embed, eps = norm_eps)

    encoder_block = EncoderBlock(
                                 self_attention = copy(attention),
                                 position_ff = copy(position_ff),
                                 norm = copy(norm),
                                 dr_rate = dr_rate)
    decoder_block = decoderBlock(
                                 self_attention = copy(attention),
                                 cross_attention = copy(attention),
                                 position_ff = copy(position_ff),
                                 norm = copy(norm),
                                 dr_rate = dr_rate)

    encoder = Encoder(
                      encoder_block = encoder_block,
                      n_layer = n_layer,
                      norm = copy(norm))
    decoder = Decoder(
                      decoder_block = decoder_block,
                      n_layer = n_layer,
                      norm = copy(norm))
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = transformer(
                        src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device

    return model