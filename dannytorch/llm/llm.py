import dannytorch.nn.nn as nn
import dannytorch.lang as lang
import numpy as np

"""
w/ Module update my require super().__init__
need_weights maybe?
figure out batch_first logic
"""
class MultiheadAttention(nn.Module):
    
    def __init__(self, d_model, n_heads, dropout=0.02, batch_first=False, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0, "number of heads must divide model dimension evenly, can't have uneven heads!!"
        self.d_model = d_model
        self.head_dim = self.d_model // n_heads
        self.n_heads = n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        #need some projections here
        self.qkv_proj  = nn.Linear(d_model, 3*d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.use_rope = use_rope
        self.pos_enc = lang.rope(self.head_dim) 
        self.attn_dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, x):
        B, T, _ = x.shape #B: batch size, T: sequence length

        qkv = self.qkv_proj(x)
        q,k,v = qkv.chunk(3, axis=-1)
        
        q = q.reshape(B, T, self.n_heads, self.head_dim)
        k = k.reshape(B, T, self.n_heads, self.head_dim)
        v = v.reshape(B, T, self.n_heads, self.head_dim)

        if self.use_rope:
            q = self.pos_enc(q, seq_len=T)
            k = self.pos_enc(k, seq_len=T)


        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)

        out = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim) #(B, H, T, T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        out = out.masked_fill(mask, -np.inf)
        attn = out.softmax()
        #dropout
        attn = self.attn_dropout(attn)
        #attention times v -> result
        res = attn @ v # (B, H, T, head_dim)
        res = res.transpose(1, 2).reshape(B, T, self.d_model) # (B, T, d_model)
        return self.o_proj(res)

class TransformerBlock(nn.Module):

    def __init__(self, d_model=128, n_heads=4, dropout=0.15):
        super().__init__()
        self.attn = MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        #self.mlp = nn.MLP() #TODO: figure out defaults, see if we can build this over sequential
        self.mlp = nn.Sequential([
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model, activation=None),
            nn.Dropout(dropout)
        ])

    def forward(self, x):
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.ln1(x)

        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)

        return x #hooray!
    
class Transformer(nn.Module):
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_blocks=6, max_seq_len=512, dropout=0.15):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = lang.rope(d_model, seq_len=max_seq_len)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size, activation=None)

    def forward(self, x):
        _, seq_len = x.shape #check for 3D

        x = self.embedding(x) * np.sqrt(self.d_model)

        #the '+=' gets messy
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