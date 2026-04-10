from dannytorch.tensor import tensor, rand
import numpy as np
from typing import OrderedDict

class Parameter:

    def __init__(self, data=None):
        self.data = tensor(data)
        self.grad = tensor(np.zeros_like(data), (self,))

    def zero_grad(self):
        self.grad = tensor(np.zeros_like(self.data))

    def __repr__(self):
        return f"Parameter({self.data})"

class Module:

    def __init__(self):
        self.training = True
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_modules', {})

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        if name in self._modules:
            return self._modules[name]
        raise AttributeError(f"No attribute found '{name}")

    def train(self, mode=True):
        self.training = mode
        for val in vars(self).values():
            if isinstance(val, Module):
                val.train(mode)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Module):
                        item.train(mode)

        return self

    def eval(self):
        return self.train(False) #recursion hehehee

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        yield from self._parameters.values()

        for module in self._modules.values():
            yield from module.parameters()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModuleList(Module):

    def __init__(self, modules=None):
        super().__init__()
        self.modules = modules or []

    def __iter__(self):
        return iter(self.modules)
    
    def __getitem__(self, idx):
        return self.modules[idx]
    
    def __len__(self):
        return len(self.modules)

    def add_module(self, module):
        self._modules.append(module)

    def append(self, module):
        self.modules.append(module)

    def extend(self, module_list):
        self.modules.extend(module_list)

    def parameters(self): 
        for m in self.modules:
            yield from m.parameters() 
    
    def train(self, mode=True):
        self.training = mode
        for module in self.modules:
            module.train(mode)

        return self

class Embedding(Module): #padding_idx is a thing

    def __init__(self, n_embeddings, embedding_dim):
        super().__init__()
        self.embedding = Parameter(np.random.randn(n_embeddings, embedding_dim))

    def forward(self, input): 
        out = tensor(self.embedding.data.data[input], (self,))

        def _backward():
            np.add.at(self.embedding.grad, input, out.grad)
        out._backward = _backward

        return out

    def parameters(self):
        return super().parameters()
    

class Linear(Module):

    #He best for relu networks
    def __init__(self, nin, nout, activation='relu', init='He'): #only alternate is Xavier, TODO: build in normal vs uniform
        # self.nodes = [Node(nin, **kwargs) for _ in range(nout)]
        super().__init__()
        self.init = init
        if self.init == 'He': self.w = Parameter(tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin)) )
        if self.init == 'Xavier': self.w = Parameter(tensor(np.random.randn(nin, nout) * np.sqrt(6.0 / (nin + nout))))
        self.b = Parameter([0 for _ in range(nout)])
        self.activation = activation
        
    def forward(self, x):
        x = x if isinstance(x, tensor) else tensor(x)
        out  = x @ self.w + self.b
        if self.activation == 'relu': out = out.relu() 
        if self.activation == 'gelu': out = out.gelu()

        return out
    
    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"Linear({self.w}, {self.b})"

class MLP(Module):

    def __init__(self, nin, nouts:list, activation='relu', init='He', dropout=0.0):
        sz = [nin] + nouts
        self.layers = ModuleList([Linear(sz[i], sz[i+1], activation= activation, init=init if i!=len(nouts)-1 else 'none') for i in range(len(nouts))])
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
        super().__init__()
        self.p = p
        self.mask = None
        
    def forward(self, x):
        if not self.training:
            return x
        
        self.mask = np.random.rand(*x.shape) > self.p #might need .astype(float)
        #we need a backward pass?
        out =  tensor(x.data * self.mask / (1-self.p))

        def backward():
            x.grad += out.grad * self.mask / (1-self.p)
        out._backward = backward

        return out

class LayerNorm(Module):
    
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Parameter(np.ones(features)) 
        self.beta = Parameter(np.zeros(features)) 

    def forward(self, x):  
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x-mean) ** 2).mean(axis=-1, keepdims=True)
        x_hat = (x-mean) / (var+self.eps) ** 0.5
        return self.gamma.data * x_hat + self.beta.data
    
    def parameters(self):
        yield self.gamma
        yield self.beta
    
class Sequential(Module):

    def __init__(self, *args):
        super().__init__()
        if isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __iter__(self):
        return iter(self._modules.values())
        
    def forward(self, x):
        
        for module in self:
            x = module(x)
        return x
    
    def append(self, module: Module):
        self.add_module(str(len(self)), module)
        return self

    def add_module(self, key, module):
        if key in self._modules:
            self._modules[key].append(module)
        else:
            self._modules[key] = [module]


    def insert(self, idx, module:Module):
        pass
    
    def extend(self, other):
        
        for layer in other:
            self.append(layer)
        return self
    
class ReLU(Module):
    def forward(self, x):
        return x.relu()

#not sure how useful this is, plus silu incomplete
class SwiGLU:
    
    def __init__(self, nin, nout):
        self.w1 = Linear(nin, nout)
        self.w2 = Linear(nin, nout)

    def forward(self, x):
        return self.w1(x).silu() * self.w2(x)


def sigmoid(input):
    return input.sigmoid()