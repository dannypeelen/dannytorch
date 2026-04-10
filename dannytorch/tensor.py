import numpy as np
from typing import List

class tensor:

    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.grad = np.zeros(self.shape)
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self.T = np.transpose(np.array(data))

    def __str__(self):
        return f"Tensor(Data:{self.data} grad:{self.grad})"

    def __repr__(self):
        return f"Tensor(Data:{self.data}, grad:{self.grad})"

    def __add__(self, other):
        other = tensor(other) if not isinstance(other, tensor) else other
        out = tensor(self.data + other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            self.grad += out.grad.reshape(self.grad.shape)
            other.grad = other.grad + out.grad.reshape(other.grad.shape)
        out._backward = _backward

        return out 

    def __mul__(self, other):
        other = tensor(other) if not isinstance(other, tensor) else other
        out = tensor(self.data * other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        other = tensor(other) if not isinstance(other, tensor) else other
        out = tensor(self.data**other.data, (self, ), self.requires_grad or other.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            self.grad += (other.data * self.data**(other.data-1)) * out.grad 
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = tensor(other) if not isinstance(other, tensor) else other
        out = tensor(self.data @ other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            
            grad_self = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            grad_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
            
            #broadcast reductions
            while grad_self.ndim > self.data.ndim:
                grad_self = grad_self.sum(axis=0)
            while grad_other.ndim > self.data.ndim:
                grad_other = grad_other.sum(axis=0)
                
            self.grad += grad_self
            other.grad += grad_other
                
            #=======2D ONLY VERSION========
            # self.grad += out.grad @ other.data.T
            # other.grad += np.atleast_2d(self.data).T @ np.atleast_2d(out.grad)
            
        out._backward = _backward

        return out
    
    def sum(self, axis=None, keepdims=False):
        out = tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), requires_grad=self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            
            grad = out.grad
            
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def exp(self):
        out = tensor(np.exp(self.data), (self,), requires_grad=self.requires_grad)
        def _backward():
            if not self.requires_grad:
                return

            self.grad += np.exp(self.data) * out.grad

        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), self.requires_grad)

        def backward():
            if not self.requires_grad:
                return
            n = self.data[..., :].size / out.data.size
            self.grad += np.ones_like(self.data) * out.grad / n
        out._backward = backward

        return out
    
    #=======activation functions==========
    def relu(self):
        out = tensor(np.maximum(0, self.data), (self,))

        def _backward():
            if not self.requires_grad:
                return
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def softmax(self):
        e_x = np.exp(self.data - np.max(self.data, keepdims=True))
        out = tensor(e_x / np.sum(e_x, keepdims=True), (self,))

        def _backward():
            # jacobian-vector product of softmax:
            if not self.requires_grad:
                return
            self.grad += out.data * (out.grad - np.dot(out.grad, out.data))
        out._backward = _backward

        return out
   
    def gelu(self):
        #chop up for future use in backwards (rip my beautiful one-liner)
        tanh_in = np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3)
        tanh_out = np.tanh(tanh_in)
        out = tensor(0.5 * self.data * (1+ tanh_out), (self,))

        def _backward():
            if not self.requires_grad:
                return
            dtanh =  (1- tanh_out ** 2)
            self.grad += out.grad * 0.5 * (1 + tanh_out) + self.data * dtanh * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.data ** 2)
        out._backward = _backward

        return out

    def sigmoid(self):
        out = tensor(1 / (1+np.exp(-self.data)), (self,))

        def _backward():
            if not self.requires_grad:
                return
            
            sig = 1 / (1+np.exp(-self.data))
            self.grad += out.grad * sig * (1-sig)

        out._backward = _backward

        return out

    def silu(self):
        sig = 1 / (1+np.exp(-self.data))
        out = tensor(self.data * sig, (self,))

        def _backward():
            if not self.requires_grad:
                return
            
            self.grad += out.grad * (self.data * sig + sig * (1 - self.data * sig))
        out._backward = _backward

        return out


    #=============end activation functions========
    #=============Tensor Operations===============

    def squeeze(self, dim=0):
        out = tensor(np.squeeze(self.data, axis=dim), (self,), self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            self.grad += np.expand_dims(out.grad, axis=dim)
        out._backward = _backward

        return out
        #new_shape = [d for d in self.shape if d != 1]
        #return self.data.reshape(new_shape)

    def unsqueeze(self, dim=0):
        out = tensor(np.expand_dims(self.data, axis=dim), (self,), self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            self.grad += np.squeeze(out.grad, axis=dim)
        out._backward = _backward

        return out

    def reshape(self, shape):
        orig_shape = self.data.shape
        out = tensor(np.reshape(self.data, shape), (self,), self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            self.grad += np.reshape(out.grad, orig_shape)
        out._backward = _backward

        return out

    def transpose(self, axes=None):
        out = tensor(np.transpose(self.data, axes=axes), (self,), self.requires_grad)
        self.T = out
        
        def _backward():
            if not self.requires_grad:
                return
            inv_axes = np.argsort(axes) if axes is not None else None
            self.grad += np.transpose(out.grad, axes=inv_axes)
        out._backward = _backward

        return out
    
    def chunk(self, chunks=2, axis=0):
        chunks = np.array_split(self.data, chunks, axis=axis)
        out = []
        
        for i, val in enumerate(chunks):
            chunk = tensor(val, (self,), requires_grad=self.requires_grad)
            start = sum(s.shape[axis] for s in chunks[:i])
            end = start + val.shape[axis]

            def _backward(idx=i, s=start, e=end):
                if not self.requires_grad:
                    return
                
                slices = [slice(None)] * self.data.ndim
                slices[axis] = slice(s, e)
                self.grad[tuple(slices)] += chunk.grad
                pass
            
            chunk._backward = _backward
            out.append(chunk)

        return out

    def cat(self, tensors: List, axis=0):
        out = tensor(np.concatenate([self.data] + [t.data for t in tensors], axis=axis), (self,), self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            
            for t in tensors:
                t.grad += out.grad
        out._backward = _backward

        return out
    
    def masked_fill(self, mask, fill_val=-np.inf):
        out = tensor(np.where(mask, fill_val, self.data), (self,), self.requires_grad)
        
        def _backward():
            if not self.requires_grad:
                return
            
            self.grad += np.where(mask, 0, out.grad)
            
        out._backward = _backward
        return out

    #=============End of TensorOps=================


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
        super().__init__(np.random.uniform(-1, 1, size=shape), _children, requires_grad)

