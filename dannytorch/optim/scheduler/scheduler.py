class StepLR:
    
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
        
    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            
class ExponentialLR(StepLR):

    def __init__(self, optimizer, gamma=0.1):
        super().__init__(optimizer, step_size=1, gamma=gamma)

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr *= self.gamma ** self.last_epoch #TODO: check formula here to confirm
        
class CosineAnnealingLR:
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def step(self):
        pass