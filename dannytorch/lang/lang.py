from dannytorch.nn.nn import Module
import numpy as np

class rope(Module):

    def __init__(self, dim, seq_len=256):
        super().__init__()
        N = 10000
        inv_freq = 1 / (N ** (np.arange(0, dim, 2).float() / dim))
        position = np.arange(seq_len).astype(float)
        sinusoid = np.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid.cos())
        self.register_buffer("sin", sinusoid.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        cos = self.cos[:seq_len].reshape(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].reshape(1, seq_len, 1, -1)
        
        return self._apply_rope(x, cos, sin)


    def _apply_rope(self, x, cos, sin):
        x1, x2 = x.chunk(2, axis=-1)
        rot_half = -x2.cat([x1], axis=-1)
        return (x*cos) + (rot_half * sin)