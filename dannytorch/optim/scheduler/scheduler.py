import numpy as np

class StepLR:
    
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
        
    def step(self): #call per epoch
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            
class ExponentialLR(StepLR):

    def __init__(self, optimizer, step_size=10, gamma=0.1):
        super().__init__(optimizer, step_size=step_size, gamma=gamma)
        self.initial_lr = self.optimizer.lr

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.initial_lr * self.gamma ** self.last_epoch

class CosineAnnealingLR:

    def __init__(self, optimizer, t_max, eta_min=0):
        self.optimizer = optimizer
        self.initial_lr = self.optimizer.lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1 + np.cos(np.pi * self.last_epoch / self.t_max))