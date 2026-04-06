import dannytorch.nn as nn
import numpy as np

class rope(nn.Module):

    def __init__(self, dim, seq_len=256):
        super().__init__()
        N = 10000
        inv_freq = 1 / (N ** (np.arange(0, dim, 2).float() / dim))
        position = np.arange(seq_len).float()
        sinusoid = 0#outer product of inv_freq and position
        self.register_buffer("cos", sinusoid.cos())
        self.register_buffer("sin", sinusoid.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)

        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        
        return self._apply_rope(x, cos, sin)


    def _apply_rope(self, x, cos, sin):
        x1, x2 = 0,0 #chunk needed
        rot_half = 0 #cat -x2,x1 at dim=-1
        return (x*cos) + (rot_half * sin)