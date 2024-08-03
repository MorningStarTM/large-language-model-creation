import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import re
from dataclasses import dataclass


device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the ModelConfig dataclass
@dataclass
class ModelConfig:
    max_iters: int = 100000
    learning_rate: float = 3e-4
    block_size: int = 8
    batch_size: int = 8
    eval_iters: int = 10000
    n_emb: int = 768
    n_layers: int = 8
    n_head: int = 8
    dropout: float = 0.1
    vocab_size: int = 50257


# Initialize the configuration
config = ModelConfig()


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_emb, head_size, bias=False)
        self.query = nn.Linear(config.n_emb, head_size, bias=False)
        self.value = nn.Linear(config.n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out




class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, config, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config.n_emb)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_emb, 4 * config.n_emb),
            nn.ReLU(),
            nn.Linear(4 * config.n_emb, config.n_emb),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_emb // config.n_head
        self.sa = MultiHeadAttention(config, config.n_head, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.ln2 = nn.LayerNorm(config.n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_emb)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_emb)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_emb)
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx