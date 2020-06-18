from tensorboardX import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def logs_training(self, reduced_losses, grad_norms, learning_rates, durations, iteration):
        for i in range(len(reduced_losses)):
            self.log_training(reduced_losses[i], grad_norms[i], learning_rates[i], durations[i],iteration+i)

    def logs_validation(self, reduced_losses, iterations):
        for i in range(1, len(reduced_losses)):
            self.log_validation(reduced_losses[i], iterations[i])

    def log_training(self, total_loss, accuracy, iteration):
        self.add_scalar("training.loss", total_loss, iteration)
        self.add_scalar("training.accuracy", accuracy, iteration)

    def log_validation(self, total_loss, accuracy, test_loss, test_accuracy, iteration):
        self.add_scalar("validation.loss", total_loss, iteration)
        self.add_scalar("validation.accuracy", accuracy, iteration)
        self.add_scalar("test_on_target.loss", test_loss, iteration)
        self.add_scalar("test_on_target.accuracy", test_accuracy, iteration)

