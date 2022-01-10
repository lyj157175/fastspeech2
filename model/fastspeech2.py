import os
import json

import torch
import torch.nn as nn

from model.transformer import Encoder, Decoder, PostNet
from .variance_adaptor import VarianceAdaptor
from my_utils.tools import get_mask

class FastSpeech2(nn.Module):

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],   # 256
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],   # 80
        )
        self.postnet = PostNet()
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r") as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(n_speaker, model_config["transformer"]["encoder_hidden"])   # n, 256

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        src_masks = get_mask(src_lens, max_src_len, device=device)
        mel_masks = None
        if mel_lens is not None:
            mel_masks = get_mask(mel_lens, max_mel_len, device=device)   #b, mel_max_len

        # b, max_len / b, max_len ---> b, max_len, enc_hidden
        output = self.encoder(texts, src_masks)
        if self.speaker_emb is not None:   # output + b, max_len, enc_hidden
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks
        ) = self.variance_adaptor(
                                    output,
                                    src_masks,
                                    mel_masks,
                                    max_mel_len,
                                    p_targets,
                                    e_targets,
                                    d_targets,
                                    p_control,
                                    e_control,
                                    d_control,
                                    device=device
                                   )
        output, mel_masks = self.decoder(output, mel_masks)   # b, mel_len, enc_hidden output为生成的mel
        output = self.mel_linear(output)    # b, mel_len, 256 ---> b, mel_len, 80
        postnet_output = self.postnet(output) + output  # b, mel_len, 80


        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
