import tensorflow as tf
import numpy as np
from Linformer_framework import LinformerEncDec
from preprocess import get_data
import hyperparameters as hp


def train(model, train_enc_lng, train_dec_lng, dec_lng_padding_index):
    data_size = train_enc_lng.shape[0]
    update_id = 0
    for epoch_id in range(hp.EPOCHS):
        print("Epoch {epoch_id}:".format(epoch_id=epoch_id))
        for i in range(0, data_size, hp.BATCH_SIZE):
            print("Update {update_id}:".format(update_id=update_id))
            enc_inputs = train_enc_lng[i:i+hp.BATCH_SIZE, :]

            batch_dec_lng = train_dec_lng[i:i+hp.BATCH_SIZE, :]
            dec_inputs = batch_dec_lng[:,:-1]
            dec_labels = batch_dec_lng[:,1:]
            
            mask = np.where(dec_labels==dec_lng_padding_index, 0, 1)

            with tf.GradientTape() as tape:
                probs = model(enc_inputs, dec_inputs) # this calls the call function conveniently
                loss = model.loss_function(probs, dec_labels, mask)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            update_id += 1


def test(model, test_enc_lng, test_dec_lng, dec_lng_padding_index):
    data_size = test_enc_lng.shape[0]
    total_acc = 0
    total_loss = 0
    word_cnt = 0

    for i in range(0, data_size, hp.BATCH_SIZE):
        # print("test", i, input_size)
        enc_inputs = test_enc_lng[i:i+hp.BATCH_SIZE, :]

        batch_dec_lng = test_dec_lng[i:i+hp.BATCH_SIZE, :]
        dec_inputs = batch_dec_lng[:,:-1]
        dec_labels = batch_dec_lng[:,1:]
        
        mask = np.where(dec_labels==dec_lng_padding_index, 0, 1)
        batch_words = np.sum(mask)

        probs = model(enc_inputs, dec_inputs) # this calls the call function conveniently
        total_loss += model.loss_function(probs, dec_labels, mask)
        # multiplied by words because we need to get total accuray per batch
        total_acc += model.accuracy_function(probs, dec_labels, mask).numpy() * batch_words

        word_cnt += batch_words

    return np.exp(total_loss/word_cnt), total_acc/word_cnt


def main(): 
    
    # data path
    enc_train = './data/train.txt'
    dec_train = './data/train.txt'
    enc_test = './data/train.txt'
    dec_test = './data/train.txt'
    # preprocess data
    train_dec, test_dec, train_enc, test_enc, dec_vocab, enc_vocab, dec_padding_index = get_data(enc_train,
                                                                                                 dec_train,
                                                                                                 enc_test,
                                                                                                 dec_test)
    # initialize the linear transformer model
    model = LinformerEncDec(    
            enc_num_tokens=len(enc_vocab),
            enc_input_size=hp.INPUT_SIZE,
            enc_channels=hp.CHANNEL,
            enc_full_attention=hp.FULL_ATTENTION,
            dec_num_tokens=len(dec_vocab),
            dec_input_size=hp.INPUT_SIZE,
            dec_channels=hp.CHANNEL,
            dec_full_attention=hp.FULL_ATTENTION,
            learning_rate=hp.LEARNING_RATE,activation="relu")

    # train the model
    train(model, train_enc, train_dec, dec_padding_index)
    # test the model
    perplexity, accuracy = test(model, test_enc, test_dec, dec_padding_index)
    print("model perplexity: {per}, accuracy: {acc}".format(per=perplexity, acc=accuracy))

if __name__ == '__main__':
    main()
