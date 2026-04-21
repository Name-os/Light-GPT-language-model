from settings import *
from data_handler import *
from model import GPTLanguageModel


#set seed for random number genrator
#this is for reporducability, no needed
if seed == None:
    seed = randint(0, 1000000000000000)
torch.manual_seed(seed)

#*******TRAINING*******#

#create the model with the vocab size
model = GPTLanguageModel(vocab_size)
#check if we train from a file
if train_from_file:
    #load the state dict
    model_state = torch.load(save_path)["model_state"]
    #set model parameters to the state dict
    model.load_state_dict(model_state)
#send the model to the correct device
model.to(device)
#create the optimizer object
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#create a function to save the model
def save():
    save_data = {
        "model_state" : model.state_dict(),
        "vocab_size"  : vocab_size,
    }

    torch.save(save_data, save_path)

print("Press 'ctrl + c' to save and stop this process")

train_estimater = utils.TrainTimeEstimater()
train_estimater.start()

try:
    for step in range(training_cycles):
        #get the batchs used for training
        questions, answers = get_batch(True)

        #evaluate the loss
        logits, loss = model(questions, answers)
        #reset the models gradents
        optimizer.zero_grad(set_to_none=True)
        #perform backpropagation
        loss.backward()
        #update the values for logits
        optimizer.step()

        if step % eval_interval == 0:
            if step != 0:
                train_estimater.stop()
                print(train_estimater.estimate(step))
                train_estimater.start()

except KeyboardInterrupt:
    save()

#save the model
save()