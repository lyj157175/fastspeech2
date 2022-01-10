import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
from my_utils.tools import get_mask, pad



class VarianceAdaptor(nn.Module):

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]  # phoneme_level
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]  # phoneme_level
        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]  # linear
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]  # linear
        n_bins = model_config["variance_embedding"]["n_bins"]    # 256

        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, n_bins - 1), requires_grad=False,)
        self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins - 1), requires_grad=False)

        self.pitch_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])
        self.energy_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])


    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)   # b, max_len - 4
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding


    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        device=torch.device("cuda"),
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)   # b, max_len - 4
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
        x = x + pitch_embedding
        energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, p_control)
        x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len, device=device)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len, device=device)
            mel_mask = get_mask(mel_len, device= device)

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len, device):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len, device):
        output, mel_len = self.LR(x, duration, max_len, device)
        return output, mel_len



class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()
        self.input_size = model_config["transformer"]["encoder_hidden"]   # 256
        self.filter_size = model_config["variance_predictor"]["filter_size"]   # 256
        self.kernel = model_config["variance_predictor"]["kernel_size"]    # 3
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]  # 256
        self.dropout = model_config["variance_predictor"]["dropout"]  # 0.3
        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1", Conv(self.input_size,
                                        self.filter_size,
                                        kernel_size=self.kernel,
                                        padding=(self.kernel - 1) // 2)),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    ("conv1d_2", Conv(self.filter_size,
                                    self.filter_size,
                                    kernel_size=self.kernel,
                                    padding=1)),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )
        self.linear_layer = nn.Linear(self.conv_output_size, 1)


    def forward(self, encoder_output, mask):   # b, max_len, enc_hidden / b, max_len
        out = self.conv_layer(encoder_output)  # b , max_len - 2 - 2, enc_hidden
        out = self.linear_layer(out)    # b, max_len - 4, 1
        out = out.squeeze(-1)  # b, max_len - 4

        if mask is not None:
            out = out.masked_fill(mask, 0.0)   #
        return out


class Conv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,   # 256
            out_channels,  # 256
            kernel_size=kernel_size,  # 3
            stride=stride,   # 1
            padding=padding,  # 1
            dilation=dilation,  # 1
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)  # b, enc_hidden, max_len
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)  # b, max_len, enc_hidden
        return x