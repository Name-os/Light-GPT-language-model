from settings import *


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