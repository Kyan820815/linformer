import tensorflow as tf
import numpy as np
from Linformer_framework import LinformerLM

"""
Example using of LinformerLM as encoder and decoder
"""

encoder = LinformerLM(
    num_tokens=10000,
    input_size=512,
    channels=16,
    dim_k=16,
    dim_ff=32,
    nhead=4,
    depth=3,
    activation="relu",
    k_reduce_by_layer=1,
    return_emb=True,
    )
decoder = LinformerLM(
    num_tokens=10000,
    input_size=512,
    channels=16,
    dim_k=16,
    dim_ff=32,
    nhead=4,
    depth=3,
    activation="relu",
    decoder_mode=True,
    )

# create batch size = 1, window size = 512 data
x = tf.convert_to_tensor(np.random.randint(1,10000,(1,512)))
y = tf.convert_to_tensor(np.random.randint(1,10000,(1,512)))

x_mask = tf.ones_like(x)==1
y_mask = tf.ones_like(y)==1

enc_output = encoder(x, input_mask=x_mask)
print(enc_output.shape) # (1, 512, 128)
dec_output = decoder(y, embeddings=enc_output, input_mask=y_mask, embeddings_mask=x_mask)
print(dec_output.shape) # (1, 512, 10000)

encoder.summary()
decoder.summary()
encoder.linformer.summary()
decoder.linformer.summary()
