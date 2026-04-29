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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#token related stuff
start_token          = "<|start|>"
end_token            = "<|end|>"

"""
1, create hyperparameters
2, create tokenizer
3, init resources and model
4, run data handler
5, train model

"""