import os
import math
import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

from .modules import BERTEmbedding, SublayerConnection, LayerNorm, PositionwiseFeedForward, MAX_VAL, MIN_VAL
from src.utils.utils import fix_random_seed_as


################
# Self-Attention
################

class Attention(nn.Module):
    def __init__(self, n=50, d=32, d_k=16, is_relative=True, args=0):
        super().__init__()
        self.n = n
        self.h = d // d_k
        self.args = args
        if self.args.local_type == "none":
            self.global_num = args.bert_num_heads
        else:
            self.global_num = args.bert_num_heads - args.local_num_heads
            self.local_num = args.local_num_heads
            assert self.h == (self.global_num + self.local_num) and self.global_num > 0

        if self.args.local_type == 'conv':
            self.convs = nn.ModuleList([self.init_conv(d_k, 2*self.args.init_val+1) for _ in range(self.local_num)])

        if self.args.local_type == 'rnn':
            self.args.init_val = int(self.args.init_val)
            position_ids_l = torch.arange(self.args.bert_max_len, dtype=torch.long).view(-1, 1)
            position_ids_r = torch.arange(self.args.init_val+1, dtype=torch.long).view(1, -1)
            self.distance = position_ids_l + position_ids_r
            self.rnns = nn.ModuleList([nn.GRU(input_size=d_k, hidden_size=d_k, num_layers=1, batch_first=True) for _ in range(self.local_num)])

        if self.args.local_type == 'win':
            position_ids_l = torch.arange(n, dtype=torch.long).view(-1, 1)
            position_ids_r = torch.arange(n, dtype=torch.long).view(1, -1)
            self.distance = (position_ids_r - position_ids_l).abs()
            self.window_size = self.args.init_val

        if self.args.local_type == 'initial':
            self.abs_pos_emb_key = nn.Embedding(n, d_k * self.local_num) 
            self.abs_pos_emb_query = nn.Embedding(n, d_k * self.local_num) 
            self.rel_pos_score = nn.Embedding(2 * n - 1, self.local_num) if is_relative else None
            sigma, alpha = 0.5, 1
            x = torch.arange(2 * n - 1) - n
            init_val = (alpha * (torch.exp(-((x/sigma)**2) / 2))).unsqueeze(-1).repeat(1, self.local_num)
            self.rel_pos_score.weight.data = init_val
            position_ids_l = torch.arange(n, dtype=torch.long).view(-1, 1)
            position_ids_r = torch.arange(n, dtype=torch.long).view(1, -1)
            self.distance = position_ids_r - position_ids_l + n - 1

        if self.args.local_type == 'adapt':
            self.abs_pos_emb_key = nn.Embedding(n, d_k * self.local_num) 
            self.abs_pos_emb_query = nn.Embedding(n, d_k * self.local_num) 
            self.rel_pos_emb = nn.Embedding(2 * n - 1, d_k * self.local_num)
            self.user_proj = nn.Linear(d, d_k * self.local_num)
            position_ids_l = torch.arange(n, dtype=torch.long).view(-1, 1)
            position_ids_r = torch.arange(n, dtype=torch.long).view(1, -1)
            self.distance = position_ids_r - position_ids_l + n - 1
            self.mlps = nn.ModuleList([nn.Linear(d_k, 1) for _ in range(self.local_num)])
            self.sigmoid = nn.Sigmoid()


    def init_conv(self, channels, kernel_size=3):
        assert (kernel_size-1) % 2 == 0
        kernel_size = int(kernel_size)
        return nn.Sequential(
            torch.nn.Conv1d(
                in_channels = channels,
                out_channels = channels,
                kernel_size = kernel_size,
                padding = (kernel_size-1) // 2
            ),
            torch.nn.ReLU()
        )

    "Compute 'Scaled Dot Product Attention"
    def forward(self, query, key, value, mask=None, dropout=None, stride=None, args=None, users=None):
        b, h, l, d_k = query.size()
        query_g, key_g, value_g = query[:, :self.global_num, ...], key[:, :self.global_num, ...], value[:, :self.global_num, ...]
        query_l, key_l, value_l = query[:, self.global_num:, ...], key[:, self.global_num:, ...], value[:, self.global_num:, ...]
        if self.args.local_type in ['initial', 'adapt']:
            index = self.distance.to(query_l.device)[-1]
            query_l = query_l + self.abs_pos_emb_query(index).unsqueeze(0).unsqueeze(0).view(1,-1,l,d_k)
            key_l = key_l + self.abs_pos_emb_key(index).unsqueeze(0).unsqueeze(0).view(1,-1,l,d_k)

        if self.global_num > 0:
            scores_g = torch.matmul(query_g, key_g.transpose(-2, -1)) / math.sqrt(query_g.size(-1))
            scores_g = scores_g.masked_fill(mask == 0, -MAX_VAL)
            p_attn_g = dropout(F.softmax(scores_g, dim=-1))
            value_g = torch.matmul(p_attn_g, value_g)

        if self.args.local_type == 'none':
            scores_l = torch.matmul(query_l, key_l.transpose(-2, -1)) / math.sqrt(query_l.size(-1))
            scores_l = scores_l.masked_fill(mask == 0, -MAX_VAL)
            p_attn_l = dropout(F.softmax(scores_l, dim=-1))
            value_l = torch.matmul(p_attn_l, value_l)

        elif self.args.local_type == 'rnn':
            value_l = torch.cat([value_l, torch.zeros(size=(b, self.local_num, self.args.init_val, d_k)).to(value_l.device)], dim=-2)
            value_aug = value_l[:, :, self.distance.to(value_l.device), :]
            h0 = torch.zeros(1, b * l * self.local_num, d_k).to(value_l.device)
            p_attn_l = None
            value_l = torch.cat([self.rnns[i](value_aug.view(-1, self.args.init_val+1, d_k), h0)[-1].view(b, self.local_num, l, d_k) for i in range(self.local_num)], dim=1)

        elif self.args.local_type == 'conv':
            p_attn_l = None
            value_l = torch.cat([self.convs[i](value_l[:, i, ...].squeeze().permute(0, 2, 1)).unsqueeze(1).permute(0,1,3,2) for i in range(self.local_num)], dim=1)

        elif self.args.local_type == 'win':
            scores_l = torch.matmul(query_l, key_l.transpose(-2, -1)) / math.sqrt(query_l.size(-1))

            mask = mask & (self.distance.to(scores_l.device) <= self.window_size)

            scores_l = scores_l.masked_fill(mask == 0, -MAX_VAL)
            p_attn_l = dropout(F.softmax(scores_l, dim=-1))
            value_l = torch.matmul(p_attn_l, value_l)

        elif self.args.local_type == 'initial' and self.rel_pos_score is not None:

            scores_l = torch.matmul(query_l, key_l.transpose(-2, -1)) / math.sqrt(query_l.size(-1))

            reweight = self.rel_pos_score(self.distance.to(scores_l.device)).unsqueeze(0).permute(0,3,1,2)
            scores_l = scores_l * (reweight / 0.1).sigmoid()
            scores_l = scores_l.masked_fill(mask == 0, -MAX_VAL)
            p_attn_l = dropout(F.softmax(scores_l, dim=-1))
            value_l = torch.matmul(p_attn_l, value_l)
            
        elif self.args.local_type == 'adapt' and self.rel_pos_emb is not None:

            scores_l = torch.matmul(query_l, key_l.transpose(-2, -1)) / math.sqrt(query_l.size(-1))

            rel_pos_embedding = self.rel_pos_emb(self.distance.to(scores_l.device)).view(l, -1, self.local_num, d_k).permute(2,0,1,3).unsqueeze(0)
            inputs = rel_pos_embedding.repeat(b,1,1,1,1) + value_l.unsqueeze(dim=-2) + value_l.unsqueeze(dim=-3) + self.user_proj(users).view(b, l, -1, d_k).permute(0,2,1,3).unsqueeze(-2)

            reweight = torch.cat([self.mlps[i](inputs[:, i, ...]).squeeze(-1).unsqueeze(1) for i in range(self.local_num)], dim=1)
            scores_l = scores_l + reweight

            scores_l = scores_l.masked_fill(mask == 0, -MAX_VAL)
            p_attn_l = dropout(F.softmax(scores_l, dim=-1))
            value_l = torch.matmul(p_attn_l, value_l)

        if self.global_num > 0:
            return torch.cat([value_g, value_l], dim=1), (p_attn_g, p_attn_l)
        else:
            return value_l, p_attn_l


class MultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, h, d_model, dropout=0.1, max_len=50, args=None):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(n=max_len, d=d_model, d_k=self.d_k, args=args)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, output=True, stride=None, args=None, users=None):

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout, stride=stride, args=args, users=users)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, att_dropout=0.2, residual=True, activate="gelu", max_len=50, args=None):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=att_dropout, max_len=max_len, args=args)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, activate=activate)

        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.residual = residual

    def forward(self, x, mask, stride=None, args=None, users=None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask, stride=stride, args=args, users=users))
        x = self.output_sublayer(x, self.feed_forward)
        return x


############
# BERT MODEL
############

class LOCKERModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        fix_random_seed_as(args.model_init_seed)

        # parameters
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        dropout = args.bert_dropout
        att_dropout = args.bert_att_dropout
        self.mask_token = args.num_items
        self.pad_token = args.num_items + 1
        self.hidden_dim = args.bert_hidden_dim

        # loss
        self.loss = nn.CrossEntropyLoss()
        self.d_loss = nn.MSELoss()

        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(vocab_size=args.num_items+2, embed_size=self.hidden_dim, max_len=args.bert_max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_dim, heads, self.hidden_dim * 4, dropout, att_dropout, max_len=args.bert_max_len, args=args) for _ in range(n_layers)])

        # weights initialization
        self.init_weights()

        # bias for similarity calculation
        self.bias = torch.nn.Embedding(num_embeddings=args.num_items+2, embedding_dim=1) # (num_items+2, 1)
        self.bias.weight.data.fill_(0)

    def forward(self, x, candidates=None, labels=None, save_name=None, users=None):

        # embedding and mask creation
        mask = (x != self.pad_token).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask_fism = ((x != self.pad_token) & (x != self.mask_token)).unsqueeze(-1)
        idx1, idx2 = self.select_predict_index(x)

        # x = self.embedding(x, is_pos=(self.args.local_type != 'soft'))
        x = self.embedding(x, is_pos=(self.args.local_type not in ['initial', 'adapt']))

        users = x.masked_fill(mask_fism==False, 0).sum(dim=-2, keepdim=True) / (mask_fism.sum(dim=-2, keepdim=True) ** 0.5)
        u = users.repeat(1, x.size(1), 1)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask, None, self.args, users=u)

        x = x[idx1, idx2]

        # similarity calculation
        logits = self.similarity_score(x, candidates)

        if labels is None:
            return logits
        else:
            labels = labels[idx1, idx2]
            loss = self.loss(logits, labels)
            return logits, loss

    def select_predict_index(self, x):
        return (x==self.mask_token).nonzero(as_tuple=True)

    def init_weights(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            # compute bounds with CDF
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            # sample uniformly from [2l-1, 2u-1] and map to normal 
            # distribution with the inverse error function
            for n, p in self.named_parameters():
                if ('norm' not in n) and ('bias' not in n) and ('rel_pos_score' not in n):
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def similarity_score(self, x, candidates):
        if candidates is None:
            w = self.embedding.token.weight.transpose(1,0)
            bias = self.bias.weight.transpose(1,0) # (1, num_items)
            return torch.matmul(x, w) + bias
        if candidates is not None:
            x = x.unsqueeze(1) # x is (batch_size, 1, embed_size)
            w = self.embedding.token(candidates).transpose(2,1) # (batch_size, embed_size, candidates)
            bias = self.bias(candidates).transpose(2,1) # (batch_size, 1, candidates)
            return (torch.bmm(x, w) + bias).squeeze(1) # (batch_size, candidates)