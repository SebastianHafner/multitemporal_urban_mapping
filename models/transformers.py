# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import copy
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from utils import experiment_manager


class TransformerModel(nn.Module):

    def __init__(self, cfg: experiment_manager.CfgNode):
        super().__init__()
        n_tokens = 3 * cfg.AUGMENTATION.CROP_SIZE * cfg.AUGMENTATION.CROP_SIZE
        d_model = 200
        d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
        n_layers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        n_heads = 2  # number of heads in nn.MultiheadAttention
        dropout = 0.2  # dropout probability

        self.model_type = 'Transformer'
        self.encoder = nn.Linear(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [T, BS, C, H, W]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = src.flatten(start_dim=2)
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)