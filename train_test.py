import tensorflow as tf
import numpy as np
import os
import pickle
from linformer_framework import LinformerEncDec
from preprocess import get_data
import hyperparameters as hp


def train(model, train_enc_lng, train_dec_lng, dec_lng_padding_index):
    data_size = train_enc_lng.shape[0]
    
    # recording # of updates
    update_id = 0
    max_update_id = int(np.ceil(data_size/hp.BATCH_SIZE*hp.EPOCHS))
    
    # error estimation
    total_acc = 0
    total_loss = 0
    word_cnt = 0
    result_list = []

    for epoch_id in range(hp.EPOCHS):
        # one update processes one batch
        for i in range(0, data_size, hp.BATCH_SIZE):
            enc_inputs = train_enc_lng[i:i+hp.BATCH_SIZE, :]

            # load batch data
            batch_dec_lng = train_dec_lng[i:i+hp.BATCH_SIZE, :]
            dec_inputs = batch_dec_lng[:,:-1]
            dec_labels = batch_dec_lng[:,1:]
            
            # ignore masking label
            mask = np.where(dec_labels==dec_lng_padding_index, 0, 1)
            batch_words = np.sum(mask)

            with tf.GradientTape() as tape:
                probs = model(enc_inputs, dec_inputs) # this calls the call function conveniently
                loss = model.loss_function(probs, dec_labels, mask)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss
            # multiplied by words because we need to get total accuray per batch
            total_acc += model.accuracy_function(probs, dec_labels, mask).numpy() * batch_words

            # compute perplexity and accuracy until now
            word_cnt += batch_words
            perplexity = np.exp(total_loss/word_cnt)
            accuracy = total_acc/word_cnt
            print("Update {update_id}/{max_update_id}: perplexity[{per}], acc[{acc}]".format(update_id=update_id, 
                                                                                             max_update_id=max_update_id,
                                                                                             per=perplexity,
                                                                                             acc=accuracy))
            result_list.append([update_id, perplexity, accuracy])
            update_id += 1
    
    return result_list


def test(model, test_enc_lng, test_dec_lng, dec_lng_padding_index):
    data_size = test_enc_lng.shape[0]
    total_acc = 0
    total_loss = 0
    word_cnt = 0

    for i in range(0, data_size, hp.BATCH_SIZE):
        enc_inputs = test_enc_lng[i:i+hp.BATCH_SIZE, :]

        # load batch data
        batch_dec_lng = test_dec_lng[i:i+hp.BATCH_SIZE, :]
        dec_inputs = batch_dec_lng[:,:-1]
        dec_labels = batch_dec_lng[:,1:]
        
        # ignore masking label
        mask = np.where(dec_labels==dec_lng_padding_index, 0, 1)
        batch_words = np.sum(mask)

        probs = model(enc_inputs, dec_inputs) # this calls the call function conveniently
        total_loss += model.loss_function(probs, dec_labels, mask)
        # multiplied by words because we need to get total accuray per batch
        total_acc += model.accuracy_function(probs, dec_labels, mask).numpy() * batch_words

        word_cnt += batch_words

    return np.exp(total_loss/word_cnt), total_acc/word_cnt


def save_result(result_list):
    """
    Save list of each {update_id, perplexity, accuracy} into file
    """
    if hp.FULL_ATTENTION == True:
        path = "{result_path}N_{input_size}_standard.txt".format(result_path=hp.RESULT_PATH, 
                                                                 input_size=hp.INPUT_SIZE)
    else:   
        path = "{result_path}N_{input_size}_k_{dim_k}.txt".format(result_path=hp.RESULT_PATH, 
                                                                  input_size=hp.INPUT_SIZE, 
                                                                  dim_k=hp.DIM_K)
    with open(path, 'wb') as fp:
        pickle.dump(result_list, fp)


def load_result():
    """
    Load list of each {update_id, perplexity, accuracy} into file
    """
    if hp.FULL_ATTENTION == True:
        path = "{result_path}N_{input_size}_standard.txt".format(result_path=hp.RESULT_PATH, 
                                                                  input_size=hp.INPUT_SIZE)
    else:
        path = "{result_path}N_{input_size}_k_{dim_k}.txt".format(result_path=hp.RESULT_PATH, 
                                                                  input_size=hp.INPUT_SIZE, 
                                                                  dim_k=hp.DIM_K)
    rlist = []
    with open(path, 'rb') as fp:
        rlist = pickle.load(fp)


def main(): 
    """
    (1) load training and test data
    (2) create model based on the settings in hyperparameter
    (3) train the model
    (4) save the result from training
    (5) test the model
    """
    # result path
    if not os.path.exists(hp.RESULT_PATH):
        os.makedirs(hp.RESULT_PATH)

    # data path
    enc_train = hp.ENC_TRAIN_PATH
    dec_train = hp.DEC_TRAIN_PATH 
    enc_test = hp.ENC_TEST_PATH
    dec_test = hp.DEC_TEST_PATH
    
    # preprocess data
    train_dec, test_dec, train_enc, test_enc, dec_vocab, enc_vocab, dec_padding_index = get_data(enc_train,
                                                                                                 dec_train,
                                                                                                 enc_test,
                                                                                                 dec_test)
    # initialize the linear transformer model
    model = LinformerEncDec(    
            enc_num_tokens=len(enc_vocab),        # encoder setting
            enc_input_size=hp.INPUT_SIZE,
            enc_channels=hp.CHANNEL,
            enc_full_attention=hp.FULL_ATTENTION,
            enc_dim_k=hp.DIM_K,
            enc_dim_d=hp.DIM_D,
            enc_dim_ff=hp.DIM_FF,
            enc_depth=hp.DEPTH,
            enc_parameter_sharing=hp.PARAMETER_SHARING,
            dec_num_tokens=len(dec_vocab),        # decoder setting
            dec_input_size=hp.INPUT_SIZE,
            dec_channels=hp.CHANNEL,
            dec_full_attention=hp.FULL_ATTENTION,
            dec_dim_k = hp.DIM_K,
            dec_dim_d = hp.DIM_D,
            dec_dim_ff=hp.DIM_FF,
            dec_depth=hp.DEPTH,
            dec_parameter_sharing=hp.PARAMETER_SHARING,
            learning_rate=hp.LEARNING_RATE,activation="relu")

    # train the model
    result_list = train(model, train_enc, train_dec, dec_padding_index)
    # save result tuple {update_id, perplexity, accuracy}
    save_result(result_list)

    # test the model
    # perplexity, accuracy = test(model, test_enc, test_dec, dec_padding_index)
    # print("model perplexity: {per}, accuracy: {acc}".format(per=perplexity, acc=accuracy))

if __name__ == '__main__':
    main()
