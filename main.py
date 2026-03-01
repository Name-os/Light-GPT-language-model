from settings import *
# import pygame

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


running = True
# show_log = False
while running:
    try:
        train_state = exists(save_path)
        dprint(f"Model state:{"" if train_state else " Not"} trained")
        choice = choose("1, Chat with model\n2, Train model\n3, Exit", ["1","2","3"])
        
        if choice == "1":
            if train_state:
                import chat
            else:
                dprint("Cannot chat, model is not trained")
        elif choice == "2":
            show_model_parameters()
            dprint("Continue with current parameters")
            # total model parameters
            # print(sum(p.numel() for p in model.parameters()), ' parameters')
            choice = choose("1, Yes\n2, No", ["1","2"])

            if choice == "1":
                import train

        elif choice == "3":
            running = False 
    except KeyboardInterrupt:
        running = False