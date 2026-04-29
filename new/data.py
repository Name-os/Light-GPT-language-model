from parameters import *

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = f.read()

chars = sorted(list(set(raw_data)))
vocab_size = len(chars)

#now to encode all of the training data into a TENSOR
all_data = torch.tensor(encode(raw_data), dtype=torch.long)

#split the data into training and evaluation in a 9 : 1 split
n = int(0.9*len(all_data))
train_data = all_data[:n]
eval_data  = all_data[n:]