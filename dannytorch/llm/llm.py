import nn.nn as nn
# import dt.lang as lang
import numpy as np

"""
w/ Module update my require super().__init__
need_weights maybe?
figure out batch_first logic
"""
class MultiheadAttention(nn.Module):
    
    def __init__(self, d_model, n_heads, dropout=0.02, batch_first=False, use_rope=True):
        assert d_model % n_heads == 0, "number of heads must divide model dimension evenly, can't have uneven heads!!"
        self.d_model = d_model
        self.head_dim = self.d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        #need some projections here

        self.use_rope = use_rope
        self.pos_enc = lang.positional_encoding() 
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        size, seq_len, _ = x.shape
        pass

class TransformerBlock(nn.Module):

    def __init__(self, d_model=128, n_heads=4, dropout=0.15):
        self.attn = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

        #self.mlp = nn.MLP() #TODO: figure out defaults, see if we can build this over sequential
        self.mlp = nn.Sequential([
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        ])

    def forward(self, x):
        attn_out = self.attn(x,x,x)
        x = x + attn_out
        x = self.ln1(x)

        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)

        return x #hooray!

class Transformer(nn.Module):
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_blocks=6, max_seq_len=512, dropout=0.15):
        #self.embedding = nn.Embedding()
        #self.pos_enc = lang.positional_encoding()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.ln = nn.LayerNorm()
        self.out_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        _, seq_len = x.shape

        x = self.embedding(x) * np.sqrt(self.d_model)

        #the '+=' gets messy
        x = x + self.pos_enc[:] #TODO: will get to logic here when I build this, also .to device is in order?
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.out_head(x)

        return logits #hooray!

#see what would the difference between this and Transformer (this might be verbose)    
class GPT(nn.Module):

    def __init__(self):
        pass
    
    
class NestedLLM(nn.Module):
    
    def __init__(self):
        pass