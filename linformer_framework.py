import tensorflow as tf
from util import identity, gen_causal_mask, get_EF, Residual, PositionalEmbedding, ProjectInOut, FeedForward
from multihead_attention import MHAttention


class Linformer(tf.keras.Model):
    """
    Reproducing the Linformer Paper
    https://arxiv.org/pdf/2006.04768.pdf

    (1) get attention for each word
    (2) apply add-normlization and residual network layer
    (3) apply feed forward network layer
    """
    def __init__(self, input_size, channels, dim_k, dim_ff=256, dim_d=None, dropout_ff=0.15,
                 nhead=4, depth=1, dropout=0.1, activation="gelu", parameter_sharing="layerwise",
                 k_reduce_by_layer=0, full_attention=False, include_ff=True, w_o_intermediate_dim=None, decoder_mode=False,
                 causal=False, method="learnable", ff_intermediate=None):
        super(Linformer, self).__init__()
        assert activation == "gelu" or activation == "relu", "Only gelu and relu activations supported for now"
        assert parameter_sharing == "none" or parameter_sharing == "headwise" or parameter_sharing == "kv" or parameter_sharing == "layerwise", "The `parameter_sharing` flag has to be either 'none', 'headwise', 'kv', or 'layerwise'."
        assert channels % nhead == 0 if dim_d is None else True, "If `dim_d` is not set to a custom value, `channels` must be divisible by `nhead`!"
        assert not (ff_intermediate and parameter_sharing=="layerwise"), "Parameter sharing must not be layerwise if ff_intermediate is enabled!"
        assert not (ff_intermediate and decoder_mode), "Raising the dimension in the middle cannot be done in the decoder!"

        layers = []
        self.decoder_mode = decoder_mode
        self.input_size = input_size
        self.channels = channels
        self.depth = depth
        self.nhead = nhead

        head_dim = channels // nhead if dim_d is None else dim_d

        # used for layer-wise parameter sharing
        E_proj = get_EF(input_size, dim_k, method, head_dim)

        causal_mask = gen_causal_mask(input_size, dim_k, full_attention) if causal else None
        # If we want causal but only with the encoder
        causal_enc = gen_causal_mask(input_size, dim_k, full_attention) if (causal and not decoder_mode) else None

        get_attn = lambda attn_channels, curr_dim_k: MHAttention(input_size, head_dim, attn_channels, curr_dim_k, nhead, dropout, parameter_sharing, E_proj, E_proj, full_attention, causal_enc, w_o_intermediate_dim, decoder_mode=False, method=method)
        get_attn_context = lambda attn_channels, curr_dim_k: MHAttention(input_size, head_dim, attn_channels, curr_dim_k, nhead, dropout, parameter_sharing, E_proj, E_proj, full_attention, causal_mask, w_o_intermediate_dim, decoder_mode=True, method=method)
        get_ff = lambda input_channels, output_channels: FeedForward(input_channels, output_channels, dim_ff, dropout_ff, activation)

        for index in range(depth):
            input_channels = ff_intermediate if (index != 0 and ff_intermediate is not None) and not decoder_mode else channels
            output_channels = ff_intermediate if (index != depth-1 and ff_intermediate is not None) and not decoder_mode else channels
            # TODO: Change the input and output channels here
            attn_layer = get_attn(input_channels, max(1, dim_k - index*k_reduce_by_layer))
            ff_layer = get_ff(input_channels, output_channels)

            attn_layer, ff_layer = map(lambda res_ch_in, res_ch_out, fn: Residual(fn, res_ch_in, res_ch_out), (input_channels, input_channels), (input_channels, output_channels), (attn_layer, ff_layer))

            if include_ff:
                layers.append(attn_layer)
                layers.append(ff_layer)
            else:
                layers.append(attn_layer)

            if not self.decoder_mode:
                continue

            attn_context = get_attn_context(channels, max(1, dim_k - index*k_reduce_by_layer))
            ff_context = get_ff(channels, channels)

            attn_context, ff_context = map(lambda fn: Residual(fn, channels, channels), (attn_context, ff_context))

            if include_ff:
                layers.append(attn_context)
                layers.append(ff_context)
            else:
                layers.append(attn_context)
        
        self.seq = layers

    def call(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len, channels)
        """
        _, n, c = tensor.shape # [batch_size, window_size, channel_size]
        assert n == self.input_size, "This tensor is of the wrong size. Dimension 1 has to match the `input_size` flag"
        assert c == self.channels, "This tensor is of the wrong size. Dimension 2 has to match the `channels` flag"
        assert "embeddings" not in kwargs or self.decoder_mode, "If decoding, needs to be initialized with `decoder_mode=True`"

        for layer in self.seq:
            tensor = layer(tensor, **kwargs)
        return tensor


class LinformerLM(tf.keras.Model):
    """
    A wrapper function to accept LM tasks, inspired by https://github.com/lucidrains/sinkhorn-transformer
    Five layer as this transformer:
    (1) embed each word in the input batch of sentences
    (2) apply positional embedding to embedding word vector
    (3) dropout layer
    (4) linear transformer layer
    (5) return logit if in decoder; otherwise, return encode_txt
    """
    def __init__(self, num_tokens, input_size, channels,
                       dim_k=64, dim_ff=1024, dim_d=None,
                       dropout_ff=0.1, dropout_tokens=0.1, nhead=4, depth=2, ff_intermediate=None,
                       dropout=0.05, activation="gelu",
                       parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False,
                       include_ff=True, w_o_intermediate_dim=None, emb_dim=None,
                       return_emb=False, decoder_mode=False, causal=False, method="learnable"):
        super(LinformerLM, self).__init__()
        emb_dim = channels if emb_dim is None else emb_dim

        self.input_size = input_size

        self.to_token_emb = tf.keras.layers.Embedding(num_tokens, emb_dim)
        self.pos_emb = PositionalEmbedding(emb_dim)
        self.dropout_tokens = tf.keras.layers.Dropout(dropout_tokens)
        self.linformer = Linformer(input_size, channels, dim_k=dim_k,
                                   dim_ff=dim_ff, dim_d=dim_d, dropout_ff=dropout_ff,
                                   nhead=nhead, depth=depth, dropout=dropout, ff_intermediate=ff_intermediate,
                                   activation=activation, parameter_sharing=parameter_sharing,
                                   k_reduce_by_layer=k_reduce_by_layer, full_attention=full_attention, include_ff=include_ff,
                                   w_o_intermediate_dim=w_o_intermediate_dim, decoder_mode=decoder_mode, causal=causal, method=method)

        if emb_dim != channels:
            self.linformer = ProjectInOut(self.linformer, emb_dim, channels)

        self.to_logits = identity if return_emb else tf.keras.layers.Dense(num_tokens, activation='softmax')

    def call(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len), and all items are ints from [0, num_tokens-1]
        """
        tensor = self.to_token_emb(tensor)
        tensor = self.pos_emb(tensor) + tensor
        tensor = self.dropout_tokens(tensor)
        tensor = self.linformer(tensor, **kwargs)
        tensor = self.to_logits(tensor)
        return tensor


class LinformerEncDec(tf.keras.Model):
    """
    A complete seq -> seq translation task. Complete with an encoder and a decoder module.
    Top Level of encoder & decder seq2seq model
    """
    def __init__(self, enc_num_tokens, enc_input_size, enc_channels, dec_num_tokens, dec_input_size, dec_channels,
                       enc_dim_k=64, enc_dim_ff=1024, enc_dim_d=None, enc_ff_intermediate=None, dec_ff_intermediate=None,
                       enc_dropout_ff=0.1, enc_nhead=4, enc_depth=2, enc_dropout=0.05, enc_parameter_sharing="layerwise", enc_k_reduce_by_layer=0,
                       enc_full_attention=False, enc_include_ff=True, enc_w_o_intermediate_dim=None, enc_emb_dim=None, enc_method="learnable",
                       dec_dim_k=64, dec_dim_ff=1024, dec_dim_d=None, dec_dropout_ff=0.1, dec_nhead=4, dec_depth=2, dec_dropout=0.05,
                       dec_parameter_sharing="layerwise", dec_k_reduce_by_layer=0, dec_full_attention=False, dec_include_ff=True,
                       dec_w_o_intermediate_dim=None, dec_emb_dim=None, dec_method="learnable", activation="gelu", learning_rate=0.001):

        super(LinformerEncDec, self).__init__()

        # optimizer and batch size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.encoder = LinformerLM(num_tokens=enc_num_tokens, input_size=enc_input_size, channels=enc_channels, dim_d=enc_dim_d, dim_ff=enc_dim_ff,
                                   dim_k=enc_dim_k, dropout_ff=enc_dropout_ff, nhead=enc_nhead, depth=enc_depth, dropout=enc_dropout,
                                   parameter_sharing=enc_parameter_sharing, k_reduce_by_layer=enc_k_reduce_by_layer, ff_intermediate=enc_ff_intermediate,
                                   full_attention=enc_full_attention, include_ff=enc_include_ff, w_o_intermediate_dim=enc_w_o_intermediate_dim,
                                   emb_dim=enc_emb_dim, return_emb=True, activation=activation, method=enc_method)
        self.decoder = LinformerLM(num_tokens=dec_num_tokens, input_size=dec_input_size, channels=dec_channels, dim_d=dec_dim_d, dim_ff=dec_dim_ff,
                                   dim_k=dec_dim_k, dropout_ff=dec_dropout_ff, nhead=dec_nhead, depth=dec_depth, dropout=dec_dropout, ff_intermediate=dec_ff_intermediate,
                                   parameter_sharing=dec_parameter_sharing, k_reduce_by_layer=dec_k_reduce_by_layer, method=dec_method,
                                   full_attention=dec_full_attention, include_ff=dec_include_ff, w_o_intermediate_dim=dec_w_o_intermediate_dim,
                                   emb_dim=dec_emb_dim, decoder_mode=True, causal=True, activation=activation)

    def call(self, x, y=None, **kwargs):
        """
        Input is (batch_size, input_size), and all items are ints from [0, num_tokens-1]
        """
        encoder_output = self.encoder(x, **kwargs)
        y = y if y is not None else x
        return self.decoder(y, embeddings=encoder_output)
    
    def accuracy_function(self, prbs, labels, mask):
        """
        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x input_size x dec_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x input_size]
        :param mask:  tensor that acts as a padding mask [batch_size x input_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))

        return accuracy

    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass

        :param prbs:  float tensor, word prediction probabilities [batch_size x input_size x dec_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x input_size]
        :param mask:  tensor that acts as a padding mask [batch_size x input_size]
        :return: the loss of the model as a tensor
        """
        prbs_masked = tf.boolean_mask(prbs, mask)
        labels_masked = tf.boolean_mask(labels, mask)
        loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels_masked, prbs_masked))

        return loss