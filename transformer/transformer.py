import torch
import math
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import Tensor


def scaled_dot_product(query, key, value, mask=None, dropout=None):
    # d_k = query.size(-1)
    d_k = query.size()[-1]
    attn_logits = torch.matmul(query, key.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    if dropout is not None:
        attention = dropout(attention)
    values = torch.matmul(attention, value)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask, dropout=self.dropout)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        """
        Inputs:
            embed_dim  - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super(EncoderBlock, self).__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoders(nn.Module):
    def __init__(self, num_layers, **block_args):
        super(TransformerEncoders, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super(CosineWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.0, max_len=50000):
#         """
#         Inputs
#             d_model - Hidden dimensionality of the input.
#             max_len - Maximum length of a sequence to expect.
#         """
#         super(PositionalEncoding, self).__init__()
#
#         self.dropout = nn.Dropout(p=dropout)
#         # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         even_div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         odd_div_term = torch.exp(torch.arange(1, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * even_div_term)
#         pe[:, 1::2] = torch.cos(position * odd_div_term)
#         pe = pe.unsqueeze(0)
#
#         # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
#         # Used for tensors that need to be on the same device as the module.
#         # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
#         self.register_buffer('pe', pe, persistent=False)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout, max_len):
#         super().__init__()
#         # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#         # max_len determines how far the position can have an effect on a token (window)
#
#         # Info
#         self.dropout = nn.Dropout(dropout)
#
#         # Encoding - From formula
#         pos_encoding = torch.zeros(max_len, d_model)
#         positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
#         division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100000.0)) / d_model) # 1000^(2i/dim_model)
#
#         # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
#         pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
#
#         # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
#         pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
#
#         # Saving buffer (same as parameter without gradients needed)
#         pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
#         self.register_buffer("pos_encoding", pos_encoding)
#
#     def forward(self, token_embedding: torch.tensor) -> torch.tensor:
#         # Residual connection + pos encoding
#         return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x