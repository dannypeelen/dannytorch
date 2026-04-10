from dannytorch.optim.scheduler.scheduler import StepLR
import numpy as np

#need to add gradient clipping?

class Adam:

    def __init__(self, params,  lr: float = 0.01, betas=[0.9, 0.999]):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.v = [0 for _ in params]
        self.s = [0 for _ in params]
        self.t = 0

    def step(self, eps=1e-8):
        self.t += 1
        for i, param in enumerate(self.params):
            grad = param.grad
            # print(f"Shape: {param.shape}, velocity: {self.v[i-1]}")
            self.v[i] = self.betas[0] * self.v[i] + (1 - self.betas[0]) * grad
            self.s[i] = self.betas[1] * self.s[i] + (1 - self.betas[1]) * (grad ** 2)
            
            #bias correction
            v_hat = self.v[i] / (1 - self.betas[0] ** self.t)
            s_hat = self.s[i] / (1- self.betas[1] ** self.t)
            
            delta_w = -self.lr * v_hat / np.sqrt(s_hat + eps)
            param.data = param.data + delta_w
            
class RMSProp:
    
    def __init__(self, params, lr: float = 0.01, betas=[0.9, 0.999]):
        self.params = params
        self.v = [0 for _ in params]
        self.betas = betas
        self.lr = lr
        self.t = 0
    
    def step(self, eps=1e-8):
        self.t += 1
        for i, param in enumerate(self.params):
            grad = param.grad
            self.v[i] = self.betas[0] * self.v[i] + (1-self.betas[0]) * grad ** 2
            #bias correction

            v_hat = self.v[i] / (1-self.betas[0] ** self.t)

            param.data = param.data - self.lr * grad / (np.sqrt(v_hat + eps))


#works only w/o momentum - same error as Adam  
class SGD:
    
    def __init__(self, params, lr=0.01, momentum=0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0 for _ in params]

    def step(self):
        if self.momentum:
            for i, param in enumerate(self.params):
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
                param.data = param.data + self.velocities[i]
        else:
            for param in self.params:
                param.data = param.data - (self.lr * param.grad)