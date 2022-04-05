import math
import numpy as np
from torch import nn
from config import *


class PositionalEncoding(nn.Module):
    """
    初始化一个positional encoding
    embed_dim: 字嵌入的维度
    max_seq_len: 最大的序列长度
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def pad_mask(seq_q, seq_k):
    """
    pad_mask 用于将填充的Padding置为True
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    return : [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_masks = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]
    pad_masks = pad_masks.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    return pad_masks


def seq_mask(seq):
    """
    seq_mask 用于decoder中屏蔽未来的单词信息
    seq: [batch_size, tgt_len]
    return : [batch_size, tgt_len, tgt_len]
    """
    seq_shape = [seq.size(0), seq.size(1), seq.size(1)]
    seq_masks = np.triu(np.ones(seq_shape), k=1)
    seq_masks = torch.from_numpy(seq_masks).byte()
    return seq_masks


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, pad_masks):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v, d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        return: output: [batch_size, n_heads, len_v, d_v]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(pad_masks, -1e9)
        weight = nn.Softmax(dim=-1)(scores)
        output = torch.matmul(weight, V) # [batch_size, n_heads, len_q, d_v]
        return output, weight


class MutiHeadAttention(nn.Module):
    def __init__(self):
        super(MutiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, pad_masks):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        return: [batch_size, len_q, d_model]
        """
        res, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        pad_masks = pad_masks.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # pad_mask : [batch_size, n_heads, seq_len, seq_len]
        context, weight = ScaledDotProductAttention()(Q, K, V, pad_masks)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model).cuda()(output + res), weight


class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, x):
        res = x
        outputs = self.model(x)
        return nn.LayerNorm(d_model).cuda()(outputs + res)
