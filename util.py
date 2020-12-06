import tensorflow as tf
import numpy as np


def identity(x, *args, **kwargs):
    return x


def get_act(activation):
    if activation == "gelu":
        return tf.nn.gelu
    if activation == "relu":
        return tf.nn.relu
    return None


def gen_causal_mask(input_size, dim_k, full_attention=False):
    """
    Generates a causal mask of size (input_size, dim_k) for linformer
    Else, it generates (input_size, input_size) for full attention
    """
    if full_attention:
        return tf.transpose(tf.linalg.band_part(tf.ones((input_size, input_size))==1, 0, -1))
    return tf.transpose(tf.linalg.band_part(tf.ones((dim_k, input_size))==1, 0, -1))


def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = tf.keras.layers.Conv1d(head_dim, head_dim, kernel_size=int(input_size/dim), stride=int(input_size/dim))
        return conv
    if method == "no_params":
        mat = tf.zeros((input_size, dim))
        tf.random.normal(mat, mean=0.0, stddev=1/dim)
        return mat
    lin = tf.keras.layers.Dense(dim, use_bias=bias)
    tf.keras.initializers.GlorotNormal(lin.weights)
    return lin


class Residual(tf.keras.layers.Layer):
    """
    Implemenation taken from
    https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
    However, I do postnorm instead of prenorm.
    """
    def __init__(self, fn, input_channels=0, output_channels=0):
        super(Residual, self).__init__()
        self.fn = fn
        self.resample = tf.keras.layers.Dense(output_channels) if input_channels != output_channels else None
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, tensor, **kwargs):
        if self.resample is not None:
            tensor = self.resample(tensor) + self.fn(tensor, **kwargs)
            tensor = self.norm(tensor)
            return tensor
        tensor = tensor + self.fn(tensor, **kwargs)
        tensor = self.norm(tensor)
        return tensor


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Standard positional embedding.
    From the paper "Attention is all you need".
    Changed the constant from 10k to 100k, since this may be better for longer sequence lengths.
    """
    def __init__(self, channels):
        super(PositionalEmbedding, self).__init__()
        self.inv_freq = 1. / (100000 ** (tf.range(0, channels, 2, dtype=float) / channels))

    def call(self, tensor):
        pos = tf.cast(tf.range(tensor.shape[1]), dtype=self.inv_freq.dtype)
        sin_inp = tf.einsum("i,j->ij", pos, self.inv_freq)
        emb = tf.concat([tf.sin(sin_inp), tf.cos(sin_inp)], axis=1)
        return emb[None, :, :]


class ProjectInOut(tf.keras.layers.Layer):
    """
    Impelemenation taken from https://github.com/lucidrains/sinkhorn-transformer/blob/73da02958965e1a690cb301292c0a3c549687d44/sinkhorn_transformer/sinkhorn_transformer.py#L218
    """
    def __init__(self, fn, dim_in, dim_out, project_out=True):
        super(ProjectInOut, self).__init__()
        self.fn = fn
        self.project_in = tf.keras.layers.Dense(dim_out)
        self.project_out = tf.keras.layers.Dense(dim_in) if project_out else identity

    def call(self, tensor, **kwargs):
        tensor = self.project_in(tensor)
        tensor = self.fn(tensor, **kwargs)
        tensor = self.project_out(tensor)
        return tensor


class FeedForward(tf.keras.layers.Layer):
    """
    Standard Feed Forward Layer
    """
    def __init__(self, input_channels, output_channels, ff_dim, dropout, activation="gelu"):
        super(FeedForward, self).__init__()
        self.w_1 = tf.keras.layers.Dense(ff_dim)
        self.w_2 = tf.keras.layers.Dense(output_channels)
        self.activation = get_act(activation)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, tensor, **kwargs):
        tensor = self.w_1(tensor)
        if self.activation is not None:
            tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)
        tensor = self.dropout2(tensor)
        return tensor
