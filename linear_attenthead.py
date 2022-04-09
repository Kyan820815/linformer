import numpy as np
import tensorflow as tf


class LinearAttentionHead(tf.keras.layers.Layer):
    """
    Linear attention, as proposed by the linformer paper
    """
    def __init__(self, dim, dropout, E_proj, F_proj, causal_mask, full_attention=False):
        super(LinearAttentionHead, self).__init__()
        """
        full_attention if true means original transformer with O(n^2) time and space complexity
        """
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.causal_mask = causal_mask
        self.is_proj_tensor = tf.is_tensor(E_proj)

    def call(self, Q, K, V, **kwargs):
        """
        Q: Q * W_q
        K: K * W_k
        V: V * W_v
        """
        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

        # masking for K, V
        if input_mask is not None:
            mask = input_mask[:,:,None]
            zero_mat = tf.zeros((mask.shape))
            K = tf.where(mask==False, zero_mat, K)
            V = tf.where(mask==False, zero_mat, V)
            del zero_mat
            del mask

        # masking for Q
        if embeddings_mask is not None:
            mask = embeddings_mask[:,:,None]
            zero_mat = tf.zeros((mask.shape))
            Q = tf.where(mask==False, zero_mat, Q)
            del zero_mat
            del mask
        
        # compute E * K * W_k if needed
        K = tf.transpose(K, perm=[0, 2, 1])
        if not self.full_attention:
            # use implementation of the paper
            if self.is_proj_tensor:
                K = tf.matmul(K, self.E)
            else:
                K = self.E(K)

        Q = tf.matmul(Q, K)

        P_bar = Q/tf.math.sqrt(float(self.dim))
        if self.causal_mask is not None:
            inf_mat = tf.cast(tf.convert_to_tensor(np.ones((self.causal_mask.shape))*np.NINF), tf.float32)
            P_bar = tf.where(self.causal_mask==False, inf_mat, P_bar)
        P_bar = tf.nn.softmax(P_bar, axis=2)

        P_bar = self.dropout(P_bar)

        # compute F * V * W_v if needed
        if not self.full_attention:
            V = tf.transpose(V, perm=[0, 2, 1])
            if self.is_proj_tensor:
                V = tf.matmul(V, self.F)
            else:
                V = self.F(V)
            V = tf.transpose(V, perm=[0, 2, 1])
        
        linear_head = tf.matmul(P_bar, V)

        return linear_head

