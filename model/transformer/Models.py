import torch
import torch.nn as nn
import numpy as np

from .Constants import PAD
from .Layers import FFTBlock
from text.symbols import my_symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        n_position = config["max_seq_len"] + 1
        src_vocab = len(my_symbols) + 1
        word_dim = config["transformer"]["encoder_hidden"]   # 256
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (config["transformer"]["encoder_hidden"] // config["transformer"]["encoder_head"])
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]   # 1000
        self.d_model = d_model
        self.src_word_emb = nn.Embedding(src_vocab, word_dim, padding_idx=PAD)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, word_dim).unsqueeze(0),
            requires_grad=False,
        )
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    # b ,max_len / b, max_len
    def forward(self, src_seq, mask, return_attns=False):
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)    # b, max_len, max_len
        enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)  # b, max_len, word_dim

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        return enc_output


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        n_position = config["max_seq_len"] + 1
        word_dim = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (config["transformer"]["decoder_hidden"] // config["transformer"]["decoder_head"])
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, word_dim).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):
        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        max_len = min(max_len, self.max_seq_len)
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        dec_output = enc_seq[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        mask = mask[:, :max_len]
        slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask
