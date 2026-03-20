import nn.nn as nn
import numpy as np

class MultiheadAttention(nn.Module):
    
    def __init__(self, embed_dim, n_heads):
        pass

    def forward(self, x):
        pass

class TransformerBlock(nn.Module):

    def __init__(self, d_model=128, n_heads=4, dropout=0.15):
        self.attn = MultiheadAttention(embed_dim=embed_dim, n_heads=n_heads)
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
        pass

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
        batch_size, seq_len = x.shape

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