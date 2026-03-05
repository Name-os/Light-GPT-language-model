import json
import time
import os
from os.path import join, exists
from random import randint


#training parameters
batch_size           = 8
block_size           = 32
num_emb_tk           = 32
num_heads            = 4
block_amount         = 4

training_cycles      = 5000
eval_interval        = 100
learning_rate        = 1e-3
dropout              = 0.1
temperature          = 0.01
seed                 = None
train_from_file      = False
training_data_amount = 5000

#paths
filter_path          = join("data", "alpaca_data.json")
save_path            = join("data", "gpt_save.pth")
data_path            = join("data", "assistiant.txt")
log_path             = join("data", "log.txt")

#other
max_tokens           = 300
target_size          = 512
show_log             = True

#token related stuff
start_token          = "<|start|>"
end_token            = "<|end|>"


import utils
log                  = utils.Log()

log.start("Loading in imports")

import torch
import torch.nn as nn
from torch.nn import functional as F
# import tokenizer

log.stop()

#check what devices can be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#*******IMPORTIANT INFORMATION*******#
"""
- Our model uses single characters as tokens instead of chuncks of words
- Token(s) - A TENSOR of a certain size that carries numerical data which corresponds to real text
             or chunks of text, or in our case, single characters
- LinearLayer - A very commonly used class which as a specified amount of neurons in a layer. Each
                neuron has a internal weight and bias (tuned during training). The weight is muliplyed
                with a input and a bias is added at the end. This is returned afterwards
- Embedding table - A type of lookup table which tokens are fed through and a result looked up. This
                    embedding tables's weights / answer to each token will be changed in training
- Overfitting - Overfitting is when the network get really good at memorising the current training
                 data but is bad a genrating. Things like dropout can be used to fix it
"""

#*******VARIABLE INFORMATION*******#
"""
device          - What device the model and related tensors runs on. It can be GPU or CPU
batch_size      - how many processes of batchs we run in paralell
block_size      - how many tokens the model can process at once, or max context length
num_emb_tk      - how much information a token carries after being looked up in the embedding table
num_heads       - how many self-attention heads we run in paralell on every tokens
training_cycles - how many batchs of data we train the model on
eval_interval   - at what interval we show the status of the model
learning_rate   - how much we step "downhill" in gradent decent
train_from_file - should we continue training from a previous state or save file
block_amount    - how many blocks the tokens go through
dropout         - precentage of neurons that are fully deactivated (precentage in decimal form)
seed            - put None for a random seed, put a number for a custom seed
max_tokens      - max amount of tokens are created during genration
save_path       - where the model weights and configurations are save
data_path       - where training data is sourced
"""

#*******HOW GPT WORKS*******#
"""
1, Import the needed training data and begin to parse it
2, All unique tokens are genrated from the data and put into a list
3, We get the total amount of tokens as this will be needed for our embedding table
4, Lookup tables are created from the vocab matching each token to a number which is used to tokenize
5, Encode and decode functions are created to encode and decode words into tokens and tokens into words
6, Then we create a get_batch function which creates training batchs for out model to use
7, Now we encode all of our raw training data into a single TENSOR
8, The TENSOR is split 9 : 1, 9 for training and 1 for evaluation
9, Now if specified, a set custom seed is set for batch making
10, The GPTLanguageModel object is created with vocab_size passed in as a parameter
11, The model creates a token embedding table, position embedding table, blocks, layernorm, and language modeling head
12, If specified, training will resume from a previous state via file
13, The model is sent to the correct device, GPU or CPU
14, Then a AdamW optimizer object is created with tbe model parameters and learning rate passed as parameters
15, 
"""