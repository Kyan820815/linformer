import tensorflow as tf
import numpy as np
from util import get_EF
from LinearAttentHead import *

class MHAttention(tf.keras.layers.Layer):
    def __init__(self, input_size, dim, channels, dim_k, nhead, dropout, checkpoint_level,
            parameter_sharing, E_proj, F_proj, full_attention, causal_mask, w_o_intermediate_dim=None, decoder_mode=False, method="learnable"):
        super(MHAttention, self).__init__()
        self.heads = [] # a list of head
        self.input_size = input_size # window_size for encoder & decoder (# of words in each batch)
        self.dim_k = dim_k # main idea of paper
        self.channels = channels # size of a word vector (before embedding)
        self.causal_mask = causal_mask 
        self.w_o_intermediate_dim = w_o_intermediate_dim
        if parameter_sharing != "layerwise":
            E_proj = get_EF(input_size, dim_k, method, dim)
            F_proj = get_EF(input_size, dim_k, method, dim) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj

        self.decoder_mode = decoder_mode
        self.to_q = []
        self.to_k = []
        self.to_v = []

        for _ in range(nhead):
            if parameter_sharing == "none":
                E_proj = get_EF(input_size, dim_k, method, dim)
                F_proj = get_EF(input_size, dim_k, method, dim)
            attn = LinearAttentionHead(dim, dropout, E_proj, F_proj, causal_mask, full_attention)
            self.heads.append(attn)
            self.to_q.append(tf.keras.layers.Dense(dim, use_bias=False))
            self.to_k.append(tf.keras.layers.Dense(dim, use_bias=False))
            self.to_v.append(tf.keras.layers.Dense(dim, use_bias=False))
        if w_o_intermediate_dim is None:
            self.w_o = tf.keras.layers.Dense(channels)
        else:
            self.w_o_1 = tf.keras.layers.Dense(w_o_intermediate_dim)
            self.w_o_2 = tf.keras.layers.Dense(channels)
        self.mh_dropout = tf.keras.layers.Dropout(dropout)

    def call(self, tensor, **kwargs):
        batch_size, input_len, channels = tensor.shape 
        assert not (self.decoder_mode and "embeddings" not in kwargs), "Embeddings must be supplied if decoding"
        assert not ("embeddings" in kwargs and (kwargs["embeddings"].shape[0], kwargs["embeddings"].shape[1], kwargs["embeddings"].shape[2]) != (batch_size, input_len, channels)), "Embeddings size must be the same as the input tensor"
        head_outputs = []
        
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor) if not self.decoder_mode else self.to_k[index](kwargs["embeddings"])
            V = self.to_v[index](tensor) if not self.decoder_mode else self.to_v[index](kwargs["embeddings"])
            head_outputs.append(head(Q,K,V,**kwargs))

        # concat all head to one macro head
        out = tf.concat(head_outputs, axis=-1)
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)
        out = self.mh_dropout(out)
        return out

