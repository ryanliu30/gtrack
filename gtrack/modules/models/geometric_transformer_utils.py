"""
Multihead Geometric Attention Implmentation
Based on labml implementation of MHA
https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/mha.py
"""

import math
from typing import Optional, List

import torch
from torch import nn
from .transformer_utils import SetNorm


class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_k: int, d_model: int, d_source: int, dropout: float = 0.1, bias: bool = True):
        """
        * `d_k` is the number of heads.
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_k
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_source, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_source, heads, d_model // heads, bias=True)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys

        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        dot_product = torch.einsum('ibhd,jbhd->ijbh', query, key)
        query_norm, key_norm = query.square().sum(-1)[:, None], key.square().sum(-1)[None, :]
        dist = torch.sqrt(0.5 * (query_norm + key_norm - 2 * dot_product).clamp(min = 1e-12))
        # Score is defined as e^(tanh(-d) + sqrt(d+1))
        scores = torch.exp(torch.tanh(-dist) - (dist + 1).sqrt())
        return scores

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has no access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape
        
        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.T[None, :, :, None], 0)

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = scores / scores.sum(1, keepdim = True).clamp(min = 1e-12)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations 
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)
    
class GeometricAttentionBlock(nn.Module):
    """
    An geometric attention block implementation based on https://nn.labml.ai/transformers/models.html
    """
    def __init__(
        self, 
        d_model: int, 
        d_ff: Optional[int] = 2048,
        d_k: Optional[int] = 3,
        heads: Optional[int] = 8, 
        dropout: Optional[float] = 0,
        d_source: Optional[int] = 512,
        self_attn: Optional[bool] = True,
        cross_attn: Optional[bool] = False,
        activation: Optional[nn.Module] = nn.LeakyReLU()
    ):
        """
        Initialize an `AttentionBlock` instance
        arguments:
            d_model: the size of input 
            d_ff: hidden size of the feed forward network
            d_k: key dimensionality
            heads: number of heads used in MHA
            dropout: dropout strength
            d_source: dimensionality of source if cross attention is used
            self_attn: whether to use self attention
            cross_attn: whether to use cross attention
            activation: activation function
        """
        super().__init__()
        if self_attn:
            self.self_attn = MultiHeadAttention(
                heads,
                d_k,
                d_model,
                d_model,
                dropout=dropout
            )
        if cross_attn:
            self.cross_attn = MultiHeadAttention(
                heads,
                d_k,
                d_model,
                d_source,
                dropout=dropout,
            ) 
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True)
        )
        self.dropout = nn.Dropout(dropout)
        if self_attn:
            self.norm_self_attn = SetNorm(d_model)
        if cross_attn:
            self.norm_cross_attn = SetNorm(d_model)
        self.norm_ff = SetNorm(d_model)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor = None,
        padding_mask: torch.Tensor = None, 
        src_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        transform the input using the attention block
        arguments:
            x: input sequence of shape (H, N, C)
            src: input source sequence of shape (S, N, C')
            padding_mask: a mask of shape (N, H) with `True` represents a 
                a fake input
            src_padding_mask: a mask of shape (N, S) with `True` represents a 
                a fake input
        returns:
            transformed sequence of shape (H, N, C)
        """
        if hasattr(self, "self_attn"):
            z = self.norm_self_attn(x, padding_mask)
            self_attn, *_ = self.self_attn(z, z, z, padding_mask)
            x = x + self.dropout(self_attn)
        if hasattr(self, "cross_attn"):
            z = self.norm_cross_attn(x, padding_mask)
            src_attn, *_ = self.cross_attn(z, src, src, src_padding_mask)
            x = x + self.dropout(src_attn)
        
        z = self.norm_ff(x, padding_mask)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        return x