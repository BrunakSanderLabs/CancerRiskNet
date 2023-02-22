import time
import os


class TimeLogger:
    """
        A helper tool for keep log of the training process with customized verbose.
        Args:
            hierachy (int): Generate with a header string of '#' for 4 x hierachy long.
                            Help to visually inspect the result log or easy grep.
            time_logger_step (int): How often we want the logger to generate an output.
                                    If logger_step is 0 (such as determined by verbose)
                                    then do not generate output.
    """
    def __init__(self, args, time_logger_step, hierachy=1, model_name="MODEL_NAME"):
        if time_logger_step == 0:
            self.logQ = False
        else:
            self.logQ = True
        self.args = args 
        self.time_logger_step = time_logger_step
        self.step_count = 0
        self.hierachy = hierachy
        self.time = time.time()
        self.model_name = model_name
        self.master_log_path = os.path.join(os.path.dirname(os.path.dirname(self.args.save_dir)), 'monitor.log')

    def log(self, s):
        if self.logQ and (self.step_count % self.time_logger_step == 0):
            print("#" * 4 * self.hierachy, " ", s, "  --time elapsed: %.2f" % (time.time() - self.time))

            with open(self.master_log_path, 'a') as f:
                f.write(" ".join([
                    self.model_name, "---", "#" * 4 * self.hierachy, s,
                    " --time elapsed: %.2f" % (time.time() - self.time), '\n'
                ]))
            f.close()
            self.time = time.time()

    def update(self):
        self.step_count += 1
        if self.logQ:
            self.log("#Refresh logger")
            self.newline()

    def newline(self):
        if self.logQ:
            print('\n')
