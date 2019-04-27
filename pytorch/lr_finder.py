import torch.optim as optim


class LRFinder:
    """Grows the learning rate exponentially.

    Based on https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(self, optimizer, max_step, init_value = 1e-8, final_value=1.):
        self.gamma = (final_value / init_value) ** (1/max_step)
        self.optimizer = optimizer
        self.init_value = init_value
        optimizer.param_groups[0]['lr'] = init_value

    def step(self, step):
        lr = self.init_value * self.gamma ** step
        self.optimizer.param_groups[0]['lr'] = lr