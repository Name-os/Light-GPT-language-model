from settings import *


log.start("Loading in data")

#extract the raw training data
with open(data_path, "r", encoding="utf-8") as file:
    raw_data = file.read()

log.stop()

log.start("Data parsing")

#this all dictates what characters the model can use to genrate
#we get a unique sorted list of all characters
chars = sorted(list(set(raw_data)))

#get the vocab size
vocab_size = len(chars)

#this makes a lookup table with each word to a number in the chars list
str_to_int = {value : index for index, value in enumerate(chars)}
#the opposite of the other table
int_to_str = {index : value for index, value in enumerate(chars)}

#create tokenizer functions
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
    
    #first its in a for loop and feeds each character one by one
    #into the lookup table and returns the corresp int
    return [str_to_int[char] for char in text]

#create decoder function
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
    
    #first its in a for loop and feeds each int one by one
    #into the lookup table and returns the corresp int
    return "".join([int_to_str[num] for num in nums])

#create the batch function which creates training batchs for the model
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

    #check which dataset to use
    data = train_data if train else eval_data

    #choose a random chunk from 0 to len of data - block_size
    #it has batch_size many sets of these random numbers
    ix = torch.randint(len(data)-block_size, (batch_size,))

    #it generates tensors of dim(batch_size, block_size)
    #this is questions which is passed into the model
    questions = torch.stack([data[i:i+block_size] for i in ix])
    #this is the answers which is used to calculate the loss function
    answers = torch.stack([data[i+1:i+block_size+1] for i in ix])

    #move the TENSORs to the correct device
    questions, answers = questions.to(device), answers.to(device)

    #return the TENSORS
    return (questions, answers)

#now to encode all of the training data into a TENSOR
all_data = torch.tensor(encode(raw_data), dtype=torch.long)

#split the data into training and evaluation in a 9 : 1 split
n = int(0.9*len(all_data))
train_data = all_data[:n]
eval_data  = all_data[n:]

log.stop()