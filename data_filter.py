from settings import *

log.start("Flitering data")

training_data = ""
cycles = 0

with open(filter_path, "r") as file:
    raw_data = json.load(file)

for question in raw_data:
    if question["input"] != "":
        continue

    # this is only suited for assistant data
    string = f"{start_token}User: {question["instruction"]}\nAssistant: {question["output"]}{end_token}\n\n\n"
    training_data += string

    cycles += 1

    if cycles == training_data_amount:
        break

with open(data_path, "w", encoding="utf-8") as file:
    file.write(training_data)

log.stop()