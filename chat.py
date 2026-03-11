from settings import *
from model import GPTLanguageModel
from data_handler import encode, decode

#load the model save data
save_data = torch.load(save_path)

#unpack the save data
model_state = save_data["model_state"]
vocab_size  = save_data["vocab_size"]

#create the model object with parameters
model = GPTLanguageModel(vocab_size)
#send the model to the correct device
model.to(device)
#load the state dict
model.load_state_dict(model_state)

#start chat
#only for testing purposes
while True:
    user_input = input("User> ")
    #create a tensor with a extra B dim (extra [])
    #also set the data type and send to device
    user_input = torch.tensor([encode("User: " + user_input)], dtype=torch.long).to(device)

    #pass tbe user input into the model along with max tokens and capture output
    #get the first index (tokens) as second is loss object and turn to list
    if True:
        output = model.generate(user_input, max_tokens)[0].tolist()
    else:
        # genrate till a certain token, end_token is used in this case
        output = model.generate_till_target(user_input, end_token)[0].tolist()


    #decode and print the output
    print(decode(output))