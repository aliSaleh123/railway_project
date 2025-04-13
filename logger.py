class Logger:
    def __init__(self):

        self.generation_number = 0      # Generation number within the overall optimization
        self.sample_number = 0          # sample number within generation

    def print(self, sample_number=0, iteration_number=0):
        print("Generation number: {}, sample number: {}, iteration number: {}".format(
            self.generation_number, sample_number, iteration_number)
              )