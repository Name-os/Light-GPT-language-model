import json
import time
from os.path import join
from os.path import join, exists
from random import randint


batch_size           = 16
block_size           = 16
num_emb_tk           = 32
num_heads            = 4
block_amount         = 4

training_cycles      = 5000
eval_interval        = 100
learning_rate        = 1e-3
dropout              = 0.1
temperature          = 0.01
seed                 = None
train_from_file      = True
training_data_amount = 5000

#home use
# filter_path          = join("Pytorch","gpt chatbot","data", "alpaca_data.json")
# save_path            = join("Pytorch","gpt chatbot","data", "gpt_save.pth")
# data_path            = join("Pytorch","gpt chatbot","data", "assistiant.txt")
# log_path             = join("Pytorch","gpt chatbot","data","log.txt")

#school use
filter_path          = join("data", "alpaca_data.json")
save_path            = join("data", "gpt_save.pth")
data_path            = join("data", "shakespeare.txt")
log_path             = join("data","log.txt")

max_tokens           = 300
target_size          = 512
show_log             = True

start_token          = "<|start|>"
end_token            = "<|end|>"


class LogingError(Exception):
    def __init__(self, message):
        super().__init__(message)

class Log:
    def __init__(self):
        self.timing = False
        self.time = 0
        self.process = ""

    def print_to_file(self, str):
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(str + "\n")
        if show_log:
            print(str)

    def log(self, str="undefined"):
        self.print_to_file(f"Aditional information to log: {str}")

    def get_time(self, return_str:bool):
        if return_str:
            return time.asctime(time.localtime(time.time()))
        return time.time_ns()

    def start(self, process="undefined"):
        if self.timing:
            raise LogingError("Class 'Log' can't log when state is True, perhaps you forgot to log off?")
        
        self.process = process
        self.time = self.get_time(False)
        self.print_to_file(f"{self.get_time(True)}, Process '{process}' has started")
        self.timing = True

    def stop(self):
        if not self.timing:
            raise LogingError("Class 'Log' can't log when state is False, perhaps you forgot to log on?")
        
        string = f"{self.get_time(True)}, Process '{self.process}' has ended with runtime of {(self.get_time(False) - self.time)*1e-9} seconds"
        self.print_to_file(string)
        self.timing = False

class TrainTimeEstimater(Log):
    def __init__(self):
        super().__init__()
        self.cycle_time = 0
    
    def start(self):
        if self.timing:
            raise LogingError("Class 'TrainTimeEstimater' can't log when state is True, perhaps you forgot to log off?")
        
        self.time = self.get_time(False)
        self.timing = True
  
    def stop(self):
        if not self.timing:
            raise LogingError("Class 'TrainTimeEstimater' can't log when state is False, perhaps you forgot to log on?")

        self.cycle_time = self.get_time(False) - self.time
        self.timing = False

    def estimate(self, step):
        if step == 0:
            step = 1

        cycles_left = training_cycles - step
        time_left = round((self.cycle_time / eval_interval) * (cycles_left / eval_interval)*0.00000006)

        h,s = divmod(time_left, 3600)
        m,s = divmod(s, 60)

        return f"Estimated time left: {h}h {m}m {s}s"

log = Log()

log.start("Loading in imports")

import torch
import torch.nn as nn
from torch.nn import functional as F

log.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AtttentionHead(nn.Module):
   """
   Class Information
   -----------------

   This is a single self-attention head, it allows tokens to talk to each other
   which encodes order making the model's prediction more accurate.

   Importiant Information
   ----------------------

   Each token emits 2 vectors; key and query.
   The key vector is what infomation the token contains.
   The query vector is what the token wants or is looking for.
   
   There are 3 Linear layers; key, query, and value:

   Key   - This layer gets all of the keys from the tokens

   Query - This layer gets all of the querys from the tokens
   
   Value - This layer output the model's prediction

   How It Works
   ------------

   1, We first feed the tokens into the key and query Linear layers to get the keys and querys.

   2, Then we take the query and key vectors and matrix multiply them to have tokens talk to each
      other using dot product. 
      
   3, We also have to make sure the results aren't too peaky, or specific values are too large or
      it can interfear with in softmax and other values will be near `0`. 

   4, This is returned into weights and masked so past tokens cant communicate
      with other past tokens that way we create a auto-regressive GPT. 

   5, Each token is asigned a value created by the value layer which will be used next.

   6, Then they are put through a softmax and matrix multiplyed with value (not layer) outputing 
      the final weights which the model will use to more betterly predict the next token.
   """
    
   def __init__(self, head_size):
      super().__init__()

      self.key = nn.Linear(num_emb_tk, head_size, bias=False)
      self.query = nn.Linear(num_emb_tk, head_size, bias=False)

      self.value = nn.Linear(num_emb_tk, head_size, bias=False)

      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

      self.dropout = nn.Dropout(dropout)

   def forward(self, tokens):
      B,T,C = tokens.shape

      key = self.key(tokens)
      query = self.query(tokens)

      weights = query @ key.transpose(-2, -1) * C**-0.5
      weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
      weights = F.softmax(weights, dim=-1)
      weights = self.dropout(weights)

      value = self.value(tokens)
      output = weights @ value

      return output

class MulitHeadAttention(nn.Module):
   """
   Class Information
   -----------------

   This performs mulit head self-attention, the reason for this is so one

   head doesn't have to be able to encode all meaning but instead many heads can

   each encode a different meaning thus deeping meaning and understanding concepts.

   This is also one of the simpler classes.

   Importiant Information
   ----------------------

   -  `nn.ModuleList` is similar to a regular python list, but pytorch cares for it.
       A good example moving devices as well as having it's state saved in state_dict (saving)

   How It Works
   ------------

   1, In the constructer it makes a certain amount of `AttentionHead` classes based on given arguments
       as well as passing head sizes into it

   2, When forward is called, it passes the given tokens into each attention head and gathers the result

   3, Lastly, it concationates the results and returns it. When it concationates the tokens, it adds more
       information so they become more expressive

   """
  
   def __init__(self, num_head, head_size):
      super().__init__()
      self.heads = nn.ModuleList([AtttentionHead(head_size) for _ in range(num_head)])
      self.projection = nn.Linear(num_emb_tk, num_emb_tk)
      self.dropout = nn.Dropout(dropout)

   def forward(self, token):
      output = torch.cat([head(token) for head in self.heads], dim=-1)
      output = self.projection(output)
      output = self.dropout(output)
      return output  

class FeedForwardLayer(nn.Module):
   """
   Class Information
   -----------------

   This is a Feed Forward Layer, it is similar to that of a MLP in shape being a

   NeuralNet but smaller. This gives tokens more time to think and deepen understanding 

   as well as adjust to the context better.

   Importiant Information
   ----------------------
   -  This is very similar to a MLP so the mechanics are very similar as well as being simple

   How It Works
   ------------
   1, It creates a Linear-Layer and a relu inside a Sequential.

   2, When forward is called, the tokens get passed through the Linear-Layer and into the relu 
      
      then returned.
   """

   def __init__(self):
      super().__init__()

      self.neural_net = nn.Sequential(
         nn.Linear(num_emb_tk, 4 * num_emb_tk),
         nn.ReLU(),
         nn.Linear(4 * num_emb_tk, num_emb_tk),

         nn.Dropout(dropout)
      )

   def forward(self, token):
      return self.neural_net(token)

class Block(nn.Module):
   """
   Class Information
   -----------------
   This class makes the training process more scaleable as the tokens are passed through

   each block and the amount of blocks can be set. Each tokens goes through these things in 

   this order; self-attention, feed forward layer, layernorm, dropout. There are also residual
   
   connection throughout the block

   Importiant Information
   ----------------------
   -  Layernorms smooth out the influence of each token so no one tokens is too powerfull

   -  Dropout deactivates some precentage of neurons each backward forward pass to ensure the
       network doesn't overfit the training data

   -  The adding of the returned tokens with the orignal tokens is called a residual pathway.
       They are used so the tokens can pile up learning and not have gradents reset each time

   How It Works
   ------------
   1, Each block first creates MultiHeadAttention, FeedForward, and 2 LayerNorm objects for
       later use

   2, Tokens are passed when called and the tokens are passed into the self-attention heads

   3, They get passed into the layernorms to get smoothed out then are returned

   4, Now they are added back to the orignal tokens to form a residual pathway

   5, The same thing is done again but with feedforward and a different layernorm

   6, The tokens are returned back into the model
   """

   def __init__(self):
      super().__init__()
      head_size = num_emb_tk // num_heads
      self.self_att_heads = MulitHeadAttention(num_heads, head_size)
      self.feed_forward = FeedForwardLayer()
      self.layer_norm_1 = nn.LayerNorm(num_emb_tk)
      self.layer_norm_2 = nn.LayerNorm(num_emb_tk)

   def forward(self, tokens):
      tokens = tokens + self.self_att_heads(self.layer_norm_1(tokens))
      tokens = tokens + self.feed_forward(self.layer_norm_2(tokens))
      return tokens


class GPTLanguageModel(nn.Module):
    """
    Class Information
    -----------------
    This is the main model, this is what makes predictions based on given text using pretrained

    weights as well as aid from self-attention heads

    Importiant Information
    ----------------------
    -   This model is different from a regular NerualNetwork, it doesn't have a neual network 
    
         but instead giant look up tables and some Linear layers.

    -   One of it's lookup tables is positional embedding where each token at position `i` gets

         a vector added to it at the `i` position on the lookup table that way more meaning is added

    -   It peforms self-attention where tokens can communiate with other tokens to better build context.

    How It Works
    ------------
    
    1, How it works is it takes in a token and looks at it's VECTOR describing it from the

        token embedding table

    2, Tokens now have shape (B,T, num_emb_tk) and each token gets a position VECTOR added to 

        it from the positional embedding table. Token `i` will get VECTOR at `i` in the table

    3, All of the tokens are passed into the self-attention head where they talk to each other
        
        and some tokens find other tokens more interesting so are further more influenced

    4, Finally, all of this is passed into a Linear layer where this is all turned into raw

        logits and returned to CrossEntropyLoss to be evaluated

    Further Information
    -------------------
    -   This is and example of a lookup table,

        `a  |  b  |  c`

        `a 0.5 | 0.9 | 0.7`
        
        `b 0.7 | 0.2 | 0.3`

        `c 0.2 | 0.1 | 0.8`

        If we index `a`, we get a VECTOR of `[[0.5, 0.9, 0.7]]`
    
    
    -   This is information on the returned TENSOR of shape (B,T,C) or similar,

        B is how many batchs we do in parallel, or batch_size

        T is the context size or block_size

        C is the raw VECTOR values for the given token

        
    -   This is information on the prediction and genration training

        The "T" bit works like this

        If we input "hell" and ask to predict the next letter, it doesn't just "o",

        instead it does:

        h -> e

        he -> l

        hel -> l

        hell -> o
        

        So this way it learns order of letters when genrating text.


    Note(s)
    -------
    -   This model genrates letter by letter so all tokens are single characters.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_emb_tk)
        self.position_embedding_table = nn.Embedding(block_size, num_emb_tk)
        self.blocks = nn.Sequential(*[Block() for _ in range(block_amount)])
        self.layer_norm = nn.LayerNorm(num_emb_tk)
        self.lang_model_head = nn.Linear(num_emb_tk, vocab_size)

    def forward(self, index, answers=None):
        _, T = index.shape
        tk_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        tokens = tk_emb + pos_emb
        tokens = self.blocks(tokens)
        tokens = self.layer_norm(tokens)
        logits = self.lang_model_head(tokens)

        if answers == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            answers = answers.view(B*T)

            loss = F.cross_entropy(logits, answers)

        return logits, loss

    def generate(self, index, new_max_tokens):
        for _ in range(new_max_tokens):
            index_crop = index[:, -block_size:]
            logits, __ = self(index_crop)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)

        return index

    def generate_till_target(self, index, target):
        while True:
            index_crop = index[:, -block_size:]
            logits, __ = self(index_crop)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
            
            if decode(index[0].tolist())[-len(target):] == target:
                break

        return index


def dprint(text, delay=0.01):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

def choose(choice_text, valid_choices:list):
    while True:
        dprint(choice_text)
        choice = input("> ")
        if choice in valid_choices:
            return choice
        dprint("Invalid choice, try again")

def show_model_parameters():
    dprint(f"Current model parameters:")
    dprint(f"Batch size: {batch_size}")
    dprint(f"Block size: {block_size}")
    dprint(f"Number of embeding tokens: {num_emb_tk}")
    dprint(f"Number of heads: {num_heads}")
    dprint(f"Number of blocks: {block_amount}")
    dprint(f"Training cycles: {training_cycles}")
    dprint(f"Learning rate: {learning_rate}")
    dprint(f"Dropout: {dropout}")
    dprint(f"Temperature: {temperature}")
    dprint(f"Seed: {"Random" if seed == None else seed}")
    dprint(f"Training from file: {train_from_file}")
    dprint(f"Training data amount: {training_data_amount}")
    dprint(f"Training file: {data_path}")
    dprint(f"Model save file location: {save_path}")

def encode(text:str):
    """
    Function
    --------
    This function is a basic tokenizer and encodes text.

    Importiant Imformation
    ----------------------
    -   This tokenizes the given text character by character as the model uses
         single characters as tokens

    Usage and parameters
    --------------------
    This function expects a string as input.

    This returns a list of numbers corresponding to their position in the full character list, 

    example is given below;

    
    full char list: `{1:a, 2:b, 3:c}`
    
    `"abcabcaaa"` -> `[1,2,3,1,2,3,1,1,1]`
    """
    
    return [str_to_int[char] for char in text]

def decode(nums:list):
    """
    Function
    --------
    This function is a basic decoder that converts tokenized information into a string.

    Importiant Imformation
    ----------------------
    None

    Usage and parameters
    --------------------
    This function expects a list of numbers as input.

    This returns a string with each number corresponding to a letter in the full character list, 
    
    example is given below;

    full char list: `{1:a, 2:b, 3:c}`
    
    `[1,2,3,1,2,3,1,1,1]` -> `"abcabcaaa"`  
    """
    
    return "".join([int_to_str[num] for num in nums])

def get_batch(train:bool):
    """
    Function
    -----
    This function creates batchs used for training and evaluation in specified sizes.

    Importiant Information
    ----------------------
    None

    Usage and Parameters
    --------------------
    This function expects a bool,
    -   If `True`, it creates batches using the training data.
    -   If `False`, it creates batches using the evaluation data.
    
    Return
    ------
    This function returns a stacked TENSOR of shape `(batch_size, batch_size, tokens)`.
    """

    data = train_data if train else eval_data
    ix = torch.randint(len(data)-block_size, (batch_size,))

    questions = torch.stack([data[i:i+block_size] for i in ix])
    answers = torch.stack([data[i+1:i+block_size+1] for i in ix])

    questions, answers = questions.to(device), answers.to(device)

    return (questions, answers)

log.start("Loading in data")
with open(data_path, "r", encoding="utf-8") as file:
    raw_data = file.read()
log.stop()
log.start("Data parsing")

chars = sorted(list(set(raw_data)))
vocab_size = len(chars)

str_to_int = {value : index for index, value in enumerate(chars)}
int_to_str = {index : value for index, value in enumerate(chars)}

all_data = torch.tensor(encode(raw_data), dtype=torch.long)
n = int(0.9*len(all_data))

train_data = all_data[:n]
eval_data  = all_data[n:]

log.stop()

running = True
while running:
    try:
        train_state = exists(save_path)
        dprint(f"Model state:{"" if train_state else " Not"} trained")
        choice = choose("1, Chat with model\n2, Train model\n3, Exit", ["1","2","3"])
        
        if choice == "1":
            if train_state:
                save_data = torch.load(save_path)

                model_state = save_data["model_state"]
                vocab_size  = save_data["vocab_size"]

                model = GPTLanguageModel(vocab_size)
                model.to(device)
                model.load_state_dict(model_state)

                while True:
                    user_input = input("User> ")
                    user_input = torch.tensor([encode("User: " + user_input)], dtype=torch.long).to(device)

                    if True:
                        output = model.generate(user_input, max_tokens)[0].tolist()
                    else:
                        output = model.generate_till_target(user_input, end_token)[0].tolist()

                    print(decode(output))
            else:
                dprint("Cannot chat, model is not trained")
        elif choice == "2":
            show_model_parameters()
            dprint("Continue with current parameters")
            # print(sum(p.numel() for p in model.parameters()), ' parameters')
            choice = choose("1, Yes\n2, No", ["1","2"])

            if choice == "1":
                if seed == None:
                    seed = randint(0, 1000000000000000)
                torch.manual_seed(seed)

                log.log(f"Seed used for batch making is: {seed}")

                log.start("Training init")

                model = GPTLanguageModel(vocab_size)
                if train_from_file:
                    model_state = torch.load(save_path)["model_state"]
                    model.load_state_dict(model_state)
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

                def save():
                    log.log("Saving model...")

                    save_data = {
                        "model_state" : model.state_dict(),
                        "vocab_size"  : vocab_size,
                    }

                    torch.save(save_data, save_path)
                    log.log(f"Model has been saved to '{save_path}'")

                log.stop()

                log.start("Training")
                log.log(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
                log.log(f"Model is {"\b" if train_from_file else "NOT"} training from a previous state")
                print("Press 'ctrl + c' to save and stop this process")

                train_estimater = TrainTimeEstimater()
                train_estimater.start()

                try:
                    for step in range(training_cycles):
                        questions, answers = get_batch(True)
                        logits, loss = model(questions, answers)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                        if step % eval_interval == 0:
                            if step != 0:
                                train_estimater.stop()
                                log.log(train_estimater.estimate(step))
                                train_estimater.start()
                            log.log(f"Training progress: {step}/{training_cycles} -> {(step/training_cycles * 100):.0f}%, Loss: {loss.item():.4f}")

                except KeyboardInterrupt:
                    log.stop()
                    log.log(f"Training paused at {log.get_time(True)}")
                    save()

                log.log("Training complete")

                save()
        elif choice == "3":
            running = False 
    except KeyboardInterrupt:
        running = False