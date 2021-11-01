import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import os

import math

MAX_VAL = 1e4
MIN_VAL = 1e-12

################
# Utils
################

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, activate="gelu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU() if activate=="gelu" else nn.ReLU()

    def forward(self, x):
        return self.w_2(self.activation(self.dropout(self.w_1(x))))

class GELU(nn.Module):
    "Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU"
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        inv = (var + self.eps).rsqrt() * self.a_2
        return x * inv + (self.b_2 - mean * inv)

class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout, residual=True):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, sublayer):
        if self.residual:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(self.dropout(sublayer(x)))

class OutputLayer(nn.Module):
    "Ouptut Layer for BERT model"
    def __init__(self, hidden_dim, activate="gelu"):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = GELU() if activate=="gelu" else nn.ReLU()
        self.layer_norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # return self.layer_norm(self.dropout(x))
        return self.layer_norm(self.activation(self.linear(x)))

################
# Embedding
################

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=vocab_size-1, )

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0):
        super().__init__()
        self.dropout = dropout
        self.max_len = max_len
        self.pe = nn.Embedding(max_len+3, d_model, padding_idx=-1) # [max_len: [sep], max_len+1: [end], max_len+2: [pad]]
        self.pe.weight.data[-1] *= 0

    def forward(self, x):
        batch_size = x.size(0)
        index = torch.arange(self.max_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        mask = torch.rand(size=index.size(), device=x.device)
        index = index.masked_fill(mask >= (1-self.dropout), -1)
        return self.pe(index)

    def more(self, x):
        index = x
        mask = torch.rand(size=index.size(), device=x.device)
        index = index.masked_fill(mask >= (1-self.dropout), -1)
        return self.pe(index)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of BERTEmbedding
    """
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, pos_dropout=0):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size, dropout=pos_dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(embed_size)
        self.embed_size = embed_size
        print('pos_dropout: ', pos_dropout)

    def forward(self, sequence, is_pos=False):
        if is_pos:
            x = self.token(sequence) + self.position(sequence)
        else:
            x = self.token(sequence)
        return self.dropout(self.norm(x))
