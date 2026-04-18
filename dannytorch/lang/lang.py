from dannytorch.nn.nn import Module
try:
    import cupy as np
except ImportError:
    import numpy as np    
from dannytorch import tensor

class rope(Module):

    def __init__(self, dim, seq_len=256):
        super().__init__()
        N = 10000
        inv_freq = 1 / (N ** (np.arange(0, dim, 2).astype(float) / dim))
        position = np.arange(seq_len).astype(float)
        sinusoid = np.outer(position, inv_freq)
        self.cos_cache = np.cos(sinusoid)
        self.sin_cache = np.sin(sinusoid)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        cos_half = self.cos_cache[:seq_len]
        sin_half = self.sin_cache[:seq_len]
        cos = np.concatenate([cos_half, cos_half], axis=-1).reshape(1, seq_len, 1, -1)
        sin = np.concatenate([sin_half, sin_half], axis=-1).reshape(1, seq_len, 1, -1)
        
        return self._apply_rope(x, cos, sin)


    def _apply_rope(self, x, cos, sin):
        x1, x2 = x.chunk(2, axis=-1)
        rot_half = -x2.cat([x1], axis=-1)
        return (x*cos) + (rot_half * sin)