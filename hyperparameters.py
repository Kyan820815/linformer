"""
Linear transformer hyperparameter settings
"""

# CHANGE TO EXPERIMENT
INPUT_SIZE = 16
DIM_K = 8

FULL_ATTENTION = False
BATCH_SIZE = 100
CHANNEL = int(2048/INPUT_SIZE) # we have a constraint that INPUT_SIZE X CHANNEL == 2048
DIM_D = int(CHANNEL/4) # we assume number of head is 4
DIM_FF = 1024
DEPTH = 1

LEARNING_RATE = 0.001
EPOCHS = 1
ENC_TRAIN_PATH = './data/train.txt'
DEC_TRAIN_PATH = './data/train.txt'
ENC_TEST_PATH = './data/train.txt'
DEC_TEST_PATH = './data/train.txt'
RESULT_PATH = './result/'