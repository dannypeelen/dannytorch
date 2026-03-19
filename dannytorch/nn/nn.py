from dannytorch.tensor import tensor, rand
import numpy as np
from typing import OrderedDict

class Module:

    def __init__(self):
        self._module = []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []
    
    #TODO:check if this is thorough
    def add_module(self, module):
        self._module.append(module)


#TODO: Build in Xavier, He, Kaiming initialization etc.
class Linear(Module):

    def __init__(self, nin, nout, nonlin=True):
        # self.nodes = [Node(nin, **kwargs) for _ in range(nout)]
        self.w = tensor(np.random.rand(nin, nout) * np.sqrt(2.0 / nin))
        self.b = tensor([0 for _ in range(nout)])
        self.nonlin = nonlin

    def __call__(self, x):
        x = x if isinstance(x, tensor) else tensor(x)
        out  = x @ self.w + self.b
        out = out.relu() if self.nonlin else out
        return out
    
    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"Linear({self.w}, {self.b})"

class MLP(Module):

    def __init__(self, nin, nouts:list):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):  
        for layer in self.layers:
            x = layer(x)

        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP({self.layers})"
        

class MSELoss:

    def __init__(self):
        pass

    def __call__(self, pred: list, expected: list) -> tensor:
        return sum((ypred-y) ** 2 for ypred, y in zip(pred, expected))



# cross entropy loss
# takes: list of raw logit tensors (shape 10,), list of integer labels
# 1. for each sample: compute softmax(logits) in numpy to get probs (shape 10,)
# 2. loss value = -mean(log(probs[true_class])) over the batch (use 1e-9 for stability)
# 3. create output tensor with _prev = tuple(preds) so backward reaches the logits
# 4. in _backward: gradient w.r.t. each logits tensor = (probs - one_hot(y)) / batch_size * out.grad
#    (this is the fused softmax+CE gradient — the jacobian collapses cleanly to this)
class CrossEntropyLoss:

    def __init__(self):
        pass

    def __call__(self, preds: list, labels) -> tensor:
        #softmax

        probs_list = []
        val_loss = 0.0

        for logits, y in zip(preds, labels):
            e_x = np.exp(logits.data - np.max(logits.data))
            prob = e_x / np.sum(e_x)
            probs_list.append(prob)
            val_loss -= np.log(prob[int(y)] + 1e-9)
        val_loss = val_loss / len(preds)

        out = tensor(np.array(val_loss), tuple(preds))

        def _backward():
            for logits, prob, y in zip(preds, probs_list, labels):
                grad = prob.copy()
                grad[int(y)] -= 1.0
                logits.grad += (grad / len(preds)) * out.grad

        out._backward = _backward

        return out
        
class Dropout: #for training, but not for inference!
    
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        
    def __call__(self, x, training=True):
        if not training:
            return x
        
        self.mask = np.random.rand(*x.shape) > self.p #might need .astype(float)
        return x * self.mask / (1-self.p)
    
class LayerNorm:
    
    def __init__(self, eps=1e-5):
        self.eps = eps
        
        
class RMSNorm:
    
    def __init__(self):
        pass
    
class Sequential(Module):

    def __init__(self, *args):
        if isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
                
    def forward(self, x):
        
        for module in self:
            x = module(x)
        return x
    
    def append(self, module: Module):
        self.add_module(str(len(self)), module)
        return self
    
    def insert(self, idx, module:Module):
        pass
    
    def extend(self, other):
        
        for layer in other:
            self.append(layer)
        return self
