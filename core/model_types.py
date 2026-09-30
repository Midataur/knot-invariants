from torch.nn import functional as F
from constants_and_types import ConfigObject, EDGES_PER_NODE
import torch.nn as nn
import torch

class Head(nn.Module):
    """
        A simple single attention head.
    """
    def __init__(self, config: ConfigObject):
        super().__init__()
        
        n_embed = config.n_embed
        n_heads = config.n_heads
        dropout = config.dropout

        head_size = n_embed // n_heads

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # this is just a place to attach a hook
        self.attention_hook = nn.Identity()
        self.sanity_hook = nn.Identity()

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        wei = self.sanity_hook(wei)

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        wei = self.attention_hook(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: ConfigObject):
        super().__init__()
        n_heads = config.n_heads
        n_embed = config.n_embed
        dropout = config.dropout

        self.heads = nn.ModuleList([Head(config) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.proj(
                torch.cat([h(x) for h in self.heads], dim=-1)
            )
        )

class FeedForward(nn.Module):
    def __init__(self, config: ConfigObject):
        super().__init__()

        n_embed, dropout = config.n_embed, config.dropout

        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Commmunication followed by computation."""

    def __init__(self, config: ConfigObject):
        super().__init__()
        n_embed = config.n_embed

        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residuals
        # don't use += as this breaks things
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BasicTransformer(nn.Module):
    """A bog standard transfomer."""
    def __init__(self, config: ConfigObject):
        super().__init__()

        self.config = config

        # derive some quantities
        context_length, vocab_size = config.get_transformer_details()

        # extract some config variables
        n_embed  = config.n_embed
        n_blocks = config.n_blocks
        proj_dim = config.proj_dim

        # create embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(context_length, n_embed)

        # this is just a place to attach a hook
        self.embed_hook = nn.Identity()
        
        # create blocks
        self.blocks = [Block(config) for _ in range(n_blocks)]
        self.blocks.append(nn.LayerNorm(n_embed))

        self.blocks = nn.Sequential(*self.blocks)

        # the projection layer.
        # applying a single MLP layer at the end improves the quality
        # of the representation the layer before this (see original SimCLR paper section 4.2).
        aggregated_dim = context_length*n_embed

        self.projection = nn.Sequential(
            nn.Linear(aggregated_dim, aggregated_dim * 4, bias=True),
            nn.ReLU(),
            nn.Linear(aggregated_dim * 4, proj_dim, bias=True),
        )

    def forward(self, input_tensor: torch.Tensor):
        batch, input_length = input_tensor.shape

        # idx and targets are both (batch, input_length) tensor of integers
        tok_emb = self.token_embedding_table(input_tensor) # (batch, input_length, embedding)
        pos_emb = self.position_embedding(torch.arange(T, device=input_tensor.device)) # (input_length, embedding)

        x = tok_emb + pos_emb # (batch, input_length, embedding)
        x = self.embed_hook(x)
        
        x = self.blocks(x) # apply a bunch of blocks (sa + feedforward) (batch, input_length, embedding)

        # reshape the matrix to be a vector
        x = x.reshape((batch, -1)) # (batch, input_length * embedding)

        # perform the projection step
        logits = self.projection(x) # (batch, proj_dim)

        return logits

MODELS = {
    "BasicTransformer": BasicTransformer
}