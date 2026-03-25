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
    
    def add_module(self, module):
        self._module.append(module)


class Embedding(Module): #padding_idx is a thing

    def __init__(self, n_embeddings, embedding_dim):
        self.embedding = np.random.randn(n_embeddings, embedding_dim)

    #TODO: figure this out
    def forward(self, input):
        return self.weigh

class Linear(Module):

    #He best for relu networks
    def __init__(self, nin, nout, activation='relu', init='He'): #only alternate is Xavier, TODO: build in normal vs uniform
        # self.nodes = [Node(nin, **kwargs) for _ in range(nout)]
        self.init = init
        if self.init == 'He': self.w = tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin)) 
        if self.init == 'Xavier': self.w = tensor(np.random.randn(nin, nout) * np.sqrt(6.0 / (nin + nout)))
        self.b = tensor([0 for _ in range(nout)])
        self.activation = activation
        
    def __call__(self, x):
        x = x if isinstance(x, tensor) else tensor(x)
        out  = x @ self.w + self.b
        if self.activation == 'relu': out = out.relu() 
        if self.activation == 'gelu': out = out.gelu() #build into tensor


        return out
    
    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"Linear({self.w}, {self.b})"

class MLP(Module):

    def __init__(self, nin, nouts:list, activation='relu', init='He', dropout=0.0):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1], activation= activation, init=init if i!=len(nouts)-1 else 'none') for i in range(len(nouts))]
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def __call__(self, x, training=True):  
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.dropout: x = self.dropout(x, training) 

        return self.layers[-1](x)
    
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
        
class Dropout(Module): #for training, but not for inference! TODO: make sure this is good w/ Module
    
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        
    def __call__(self, x, training=True):
        if not training:
            return x
        
        self.mask = np.random.rand(*x.shape) > self.p #might need .astype(float)
        return x * self.mask / (1-self.p)
    
class LayerNorm(Module): #TODO: check, see how to do mean and var, fit in with autograd
    
    def __init__(self, features, eps=1e-5):
        # super().__init__()
        self.eps = eps
        self.gamma = 0 #this has to be nn.Paramter, worth building?
        self.beta = 0 #^

    def forward(self, x):  
        mean = x.mean()
        var = x.var()
        return self.gamma * (x-mean) / np.sqrt(var+self.eps) + self.beta

        
        
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

    def __iter__(self):
        pass
        
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
    
def ReLU(input):
    return input.relu()

#not sure how useful this is, plus silu incomplete
class SwiGLU(Module):
    
    def __init__(self, nin, nout):
        super().__init__()
        self.w1 = Linear(nin, nout)
        self.w2 = Linear(nin, nout)

    def forward(self, x):
        return self.w1(x).silu() * self.w2(x)


def sigmoid(input):
    pass