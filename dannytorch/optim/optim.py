from scheduler.scheduler import StepLR

class Adam:

    def __init__(self, params,  lr: float | StepLR = 0.01, betas=[0.9, 0.999]):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.v = [0 for _ in params]
        self.s = [0 for _ in params]
        self.t = 0

    def step(self, eps=1e-8):
        self.t += 1
        #TODO: figure out fallback here
        for i, param in enumerate(self.params):
            grad = param.grad
            self.v[i] = self.betas[0] * self.v[i-1] + (1 - self.betas[0]) * grad
            self.s[i] = self.betas[1] * self.s[i-1] + (1 - self.betas[1]) * (grad ** 2)
            
            #bias correction
            v_hat = self.v[i] / (1 - self.betas[0])
            s_hat = self.s[i] / (1- self.betas[1])
            
            delta_w = -self.lr * (v_hat / np.sqrt(s_hat + eps) * grad)
            param.data += delta_w
            
class RMSProp:
    
    def __init__(self):
        pass
    
    def step(self):
        pass #TODO:this is very similar to Adam
    
    
class SGD:
    
    def __init__(self, params, lr=0.01, momentum=0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0 for _ in params]

    def step(self):
        if self.momentum:
            #TODO: figure out fallback here for v_0, maybe keep running variable of v_prev
            for i, param in enumerate(self.params):
                if i == 0:
                    self.velocities[i] = -self.lr * param.grad
                else:
                    self.velocities[i] = self.momentum * self.velocities[i-1] - self.lr * param.grad
                param.data += self.velocities[i]
        else:
            for param in self.params:
                param.data -= self.lr * param.grad