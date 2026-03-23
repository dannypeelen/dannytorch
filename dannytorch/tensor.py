import numpy as np

class tensor:

    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.grad = np.zeros(self.shape)
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self._prev = set(_children)

    def __str__(self):
        return f"Tensor(Data:{self.data} grad:{self.grad})"

    def __repr__(self):
        return f"Tensor(Data:{self.data}, grad:{self.grad})"

    def __add__(self, other):
        other = tensor(other) if not isinstance(other, tensor) else other
        out = tensor(self.data + other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward():
            self.grad += out.grad.reshape(self.grad.shape)
            other.grad = other.grad + out.grad.reshape(other.grad.shape)
        out._backward = _backward

        return out 

    def __mul__(self, other):
        other = tensor(other) if not isinstance(other, tensor) else other
        out = tensor(self.data * other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        other = tensor(other) if not isinstance(other, tensor) else other
        out = tensor(self.data**other.data, (self, ), self.requires_grad or other.requires_grad)

        def _backward():
            self.grad += (other.data * self.data**(other.data-1)) * out.grad 
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = tensor(other) if not isinstance(other, tensor) else other
        out = tensor(self.data @ other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += np.atleast_2d(self.data).T @ np.atleast_2d(out.grad)
        out._backward = _backward

        return out
    
    #=======activation functions==========
    def relu(self):
        out = tensor(np.maximum(0, self.data), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def softmax(self):
        e_x = np.exp(self.data - np.max(self.data, keepdims=True))
        out = e_x / np.sum(e_x)

        def _backward():
            # jacobian-vector product of softmax:
            self.grad += out.data * (out.grad - np.dot(out.grad, out.data))
        out._backward = _backward

        return out
   
    def gelu(self): #TODO: just do whenever
        pass

    def sigmoid(self):
        pass

    #TODO: computer back pass math
    def silu(self):
        out = tensor(self.data / (1+ np.exp(-self.data)))

        def _backward():
            self.grad += 1
        out._backward = _backward

        return out


    #=============end activation functions========

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self * other
    
    def __rmatmul__(self, other):
        return self @ other
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def backward(self):
        #topo sort done here

        #run through topo sort and call backward
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


class rand(tensor):

    def __init__(self, shape=1, requires_grad=True, _children=None):
        super().__init__(np.random.uniform(-1, 1, size=shape), requires_grad, _children)