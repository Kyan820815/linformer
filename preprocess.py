import tensorflow as tf
import numpy as np
import hyperparameters as hp


##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
##########DO NOT CHANGE#####################

def pad_corpus(encode_txt, decode_txt):
    """
    :param encode_txt: list of ENCODE sentences
    :param decode_txt: list of DECODE sentences
    :return: A tuple of: (list of padded sentences for ENCODE, list of padded sentences for DECODE)
    """
    encode_padded_sentences = []

    for line in encode_txt:
        if len(line) == 0:
            continue
        padded_encode = line[:hp.INPUT_SIZE-1]
        padded_encode += [STOP_TOKEN] + [PAD_TOKEN] * (hp.INPUT_SIZE - len(padded_encode)-1)
        encode_padded_sentences.append(padded_encode)

    decode_padded_sentences = []
    for line in decode_txt:
        if len(line) == 0:
            continue
        padded_decode = line[:hp.INPUT_SIZE-1]
        padded_decode = [START_TOKEN] + padded_decode + [STOP_TOKEN] + [PAD_TOKEN] * (hp.INPUT_SIZE - len(padded_decode)-1)
        decode_padded_sentences.append(padded_decode)

    return encode_padded_sentences, decode_padded_sentences


def build_vocab(sentences):
    """
    :param sentences:  list of sentences, each a list of words
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

    vocab =  {word:i for i,word in enumerate(all_words)}

    return vocab,vocab[PAD_TOKEN]


def convert_to_id(vocab, sentences):
    """
    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
    """
    :param file_name:  string, name of data file
    :return: list of sentences, each a list of words split on whitespace
    """
    text = []
    with open(file_name, 'rt', encoding='latin') as data_file:
        for line in data_file: text.append(line.split())
    return text


def get_data(enc_training_file, dec_training_file, enc_test_file, dec_test_file):
    """
    Use the helper functions in this file to read and parse training and test data, then pad the corpus.
    Then vectorize your train and test data based on your vocabulary dictionaries.

    :param enc_training_file: Path to the ENCODE training file.
    :param dec_training_file: Path to the DECODE training file.
    :param enc_test_file: Path to the ENCODE test file.
    :param dec_test_file: Path to the DECODE test file.
    
    :return: Tuple of train containing:
    (2-d list or array with DECODE TRAINING sentences in vectorized/id form [num_sentences x input_size+1] ),
    (2-d list or array with DECODE TEST sentences in vectorized/id form [num_sentences x input_size+1]),
    (2-d list or array with ENCODE TRAINING sentences in vectorized/id form [num_sentences x input_size]),
    (2-d list or array with ENCODE TEST sentences in vectorized/id form [num_sentences x input_size]),
    DECODE VOCAB (Dict containg word->index mapping),
    ENCODE VOCAB (Dict containg word->index mapping),
    DECODE padding ID (the ID used for *PAD* in the decode vocab. This will be used for masking loss)
    """

    #1) Read encode and decode Data for training and testing (see read_data)
    enc_train = read_data(enc_training_file)
    dec_train = read_data(dec_training_file)
    enc_test = read_data(enc_test_file)
    dec_test = read_data(dec_test_file)

    #2) Pad training data (see pad_corpus)
    enc_train_pad, dec_train_pad = pad_corpus(enc_train, dec_train)

    #3) Pad testing data (see pad_corpus)
    enc_test_pad, dec_test_pad = pad_corpus(enc_test, dec_test)

    #4) Build vocab for encode (see build_vocab)
    enc_vocab, enc_pt_id = build_vocab(enc_train_pad)

    #5) Build vocab for decode (see build_vocab)
    dec_vocab, dec_pt_id = build_vocab(dec_train_pad)

    #6) Convert training and testing encode sentences to list of IDS (see convert_to_id)
    enc_train_id = convert_to_id(enc_vocab, enc_train_pad)
    enc_test_id = convert_to_id(enc_vocab, enc_test_pad)

    #7) Convert training and testing decode sentences to list of IDS (see convert_to_id)
    dec_train_id = convert_to_id(dec_vocab, dec_train_pad)
    dec_test_id = convert_to_id(dec_vocab, dec_test_pad)

    return  np.array(dec_train_id, dtype=np.int32), \
            np.array(dec_test_id, dtype=np.int32), \
            np.array(enc_train_id, dtype=np.int32), \
            np.array(enc_test_id, dtype=np.int32), \
            dec_vocab, enc_vocab, dec_pt_id


    