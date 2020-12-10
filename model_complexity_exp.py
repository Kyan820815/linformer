import tensorflow as tf
from datetime import datetime
from multihead_attention import MHAttention
from util import get_EF

"""
We test how the relation between input size (also know as window size) and the parameter dim_k proposed by the paper
For each input size, we first test vanilla transfomer, which is O(n^2). Then we will test different dim_k used in
linaer transformer
"""


def create_model(input_size, k, n_head, E_proj=None, F_proj=None, full_attention=True):
    model = MHAttention(
                        input_size=input_size, # Dimension 1 of the input
                        channels=64,   # Dimension 2 of the input
                        dim=64/n_head, # Dim of each attn head
                        dim_k=k,       # What to sample the input length down to
                        nhead=n_head,  # Number of heads
                        dropout=0,     # Dropout for each of the heads
                        causal_mask=None,
                        parameter_sharing="layerwise", # What level of parameter sharing to do
                        E_proj=E_proj, F_proj=F_proj,  # The E and F projection matrices
                        full_attention=full_attention, # Use full attention instead
                        )
    return model


def main():
    # input size
    ns = [256, 512, 1024, 2048, 4096, 8192, 16384]
    # dim k
    ks = [128, 256, 512, 1024, 2048]
    batch_size = 1
    iteration = 1

    for n in ns:
        model = create_model(n, n, 4)
        # create input data
        x = tf.random.normal([batch_size, n, 64])

        # test vanilla transformer performance
        start = datetime.now()
        for _ in range(iteration):
            y = model(x)
        end = datetime.now()
        base_time = (end - start).microseconds
        print("======================================================")
        for k in ks:
            if n > k:
                E_proj = get_EF(n, k)
                F_proj = get_EF(n, k)
                model = create_model(n, k, 1, E_proj, F_proj, False)
                # create input data
                x = tf.random.normal([batch_size, n, 64])

                # test linear transformer performance
                start = datetime.now()
                for _ in range(iteration):
                    y = model(x)
                end = datetime.now()
                curr_time = (end - start).microseconds
                print("n: ", n, ", k: ",k , " improve: ", base_time / curr_time)

if __name__ == '__main__':
    main()
