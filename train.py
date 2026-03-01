from settings import *
from data_handler import *
from model import GPTLanguageModel


#set seed for random number genrator
#this is for reporducability, no needed
if seed == None:
    seed = randint(0, 1000000000000000)
torch.manual_seed(seed)

log.log(f"Seed used for batch making is: {seed}")

log.start("Training init")

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
                log.log(train_estimater.estimate(step))
                train_estimater.start()
            log.log(f"Training progress: {step}/{training_cycles} -> {(step/training_cycles * 100):.0f}%, Loss: {loss.item():.4f}")

except KeyboardInterrupt:
    log.stop()
    log.log(f"Training paused at {log.get_time(True)}")
    save()

log.log("Training complete")

#save the model
save()