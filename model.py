#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    Multi-head attention mechanism module.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        block_size (int): Block size for the attention mechanism.
        dropout (float, optional): Dropout probability. Default is 0.0.
    """

    def __init__(self, in_features: int, out_features: int, block_size: int, dropout: float = 0.0):
        """
        Initializes the Head module.

        Initializes linear layers for key, query, and value, and other parameters.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            block_size (int): Block size for the attention mechanism.
            dropout (float, optional): Dropout probability. Default is 0.0.
        """
        super().__init__()
        # Linear layers for key, query, and value
        self.key = nn.Linear(in_features, out_features, bias=False)
        self.query = nn.Linear(in_features, out_features, bias=False)
        self.value = nn.Linear(in_features, out_features, bias=False)
        # Register buffer for lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self._norm = math.sqrt(out_features)  # Normalization factor
        self.att_dropout = nn.Dropout(dropout)  # Attention dropout
        self.res_dropout = nn.Dropout(dropout)  # Residual dropout
        # Linear projection layer
        self.proj = nn.Linear(out_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Computes multi-head attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor after multi-head attention mechanism.
        """
        B, T, E = x.shape  # Batch size, sequence length, embedding dimension

        k = self.key(x)    # Compute key
        q = self.query(x)  # Compute query
        v = self.value(x)  # Compute value

        attn = q @ k.transpose(-1, -2) / self._norm  # Compute attention scores
        # Mask attention scores with lower triangular matrix
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)  # Apply softmax
        attn = self.att_dropout(attn)  # Apply attention dropout

        out = attn @ v  # Compute output
        out = self.proj(out)  # Linear projection
        out = self.res_dropout(out)  # Apply residual dropout

        return out




class MultiHeadAttentionNaive(nn.Module):
  """
  Implements a naive version of multi-head attention by creating multiple
  'Head' instances and concatenating their outputs. This approach is straightforward
  but less efficient than combining operations across heads.
  """
  def __init__(self, num_heads: int, in_features: int, out_features: int, block_size: int, dropout: float = 0.0) -> None:
    super().__init__()
    # Initialize a ModuleList of 'Head' modules, each representing an attention head.
    self.heads = nn.ModuleList([
      Head(in_features=in_features, out_features=out_features//num_heads, block_size=block_size, dropout=dropout)
      for _ in range(num_heads)
    ])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Concatenate the outputs of all heads along the last dimension.
    return torch.concat([h(x) for h in self.heads], dim=-1)



# File: multi_head_attention_module.py

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism module.

    Args:
        num_heads (int): Number of attention heads.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        block_size (int): Block size for the attention mechanism.
        dropout (float, optional): Dropout probability. Default is 0.0.
    """

    def __init__(self, num_heads: int, in_features: int, out_features: int, block_size: int, dropout: float = 0.0) -> None:
        """
        Initializes the MultiHeadAttention module.

        Initializes linear layers for query, key, and value, and other parameters.

        Args:
            num_heads (int): Number of attention heads.
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            block_size (int): Block size for the attention mechanism.
            dropout (float, optional): Dropout probability. Default is 0.0.
        """
        super().__init__()
        assert in_features % num_heads == 0

        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.query_key_value = nn.Linear(in_features, out_features * 3, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self._norm = math.sqrt(out_features)
        self.att_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(out_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Computes multi-head attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor after multi-head attention mechanism.
        """
        B, T, _ = x.shape
        q, k, v = self.query_key_value(x).chunk(3, dim=-1)

        # Original calculation (naive):
        # q = q.view(B, T, self.num_heads, self.out_features // self.num_heads).transpose(1, 2)
        # k = k.view(B, T, self.num_heads, self.out_features // self.num_heads).transpose(1, 2)
        # v = v.view(B, T, self.num_heads, self.out_features // self.num_heads).transpose(1, 2)

        # Combined head calculation (improved):
        q = k = v = self._combine_heads(q, k, v)

        # Remaining steps of MultiHeadAttention are unchanged

    def _combine_heads(self, q, k, v):
        """
        Combines query, key, and value tensors for multi-head attention.

        Reshapes and transposes the tensors for multi-head attention computation.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Combined query, key, and value tensors for multi-head attention.
        """
        # Reshape for multi-head attention: split the last dimension into (num_heads, depth)
        B, T, _ = q.shape
        depth = self.out_features // self.num_heads
        q = q.view(B, T, self.num_heads, depth).transpose(1, 2)  # B, num_heads, T, depth
        k = k.view(B, T, self.num_heads, depth).transpose(1, 2)
        v = v.view(B, T, self.num_heads, depth).transpose(1, 2)
        # No need to combine q, k, v here; they will be processed in their respective heads
        return q, k, v


class FeedForward(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features // 4),
            nn.ReLU(),
            nn.Linear(in_features=out_features // 4, out_features=out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Represents a single Transformer block, which includes a multi-head attention
    mechanism, followed by layer normalization and a feed-forward network, with
    residual connections around both the attention and feed-forward modules.
    """
    def __init__(self, n_embed: int, n_head: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Multi-head attention module
        self.attn = MultiHeadAttention(n_head, n_embed, n_embed, block_size, dropout=dropout)
        # Layer normalization before the attention and feed-forward networks
        self.ln1 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, n_embed, dropout=dropout)  # Feed-forward network
        self.ln2 = nn.LayerNorm(n_embed)  # Layer normalization after the feed-forward network

    def forward(self, x):
        # Apply multi-head attention with a residual connection
        x = x + self.attn(self.ln1(x))
        # Apply feed-forward network with a residual connection
        x = x + self.ffn(self.ln2(x))
        return x


class BigramLangModel(nn.Module):
    """
    Bigram language model based on Transformer architecture.

    Args:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Block size for positional embeddings.
        embedding_size (int, optional): Size of the token embeddings. Default is 128.
        depth (int, optional): Number of transformer blocks. Default is 4.
        num_heads (int, optional): Number of attention heads. Default is 4.
        dropout (float, optional): Dropout probability. Default is 0.0.
    """

    def __init__(self, vocab_size: int, block_size: int, embedding_size: int = 128, depth: int = 4, num_heads: int = 4, dropout: float = 0.0):
        """
        Initializes the BigramLangModel module.

        Initializes embedding layers, transformer blocks, layer normalization, and linear head.

        Args:
            vocab_size (int): Size of the vocabulary.
            block_size (int): Block size for positional embeddings.
            embedding_size (int, optional): Size of the token embeddings. Default is 128.
            depth (int, optional): Number of transformer blocks. Default is 4.
            num_heads (int, optional): Number of attention heads. Default is 4.
            dropout (float, optional): Dropout probability. Default is 0.0.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.depth = depth
        self.num_heads = num_heads

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_size, num_heads, block_size, dropout=dropout)
            for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def hyper_params(self):
        """
        Returns the hyperparameters of the model.

        Returns:
            dict: Dictionary containing the hyperparameters.
        """
        return {
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'embedding_size': self.embedding_size,
            'depth': self.depth,
            'num_heads': self.num_heads,
        }

    @property
    def device(self):
        """
        Returns the device of the model.

        Returns:
            torch.device: Device of the model.
        """
        return self.token_embedding_table.weight.device


    def forward(self, inputs: torch.Tensor):
        # inputs: (batch_size, block_size)
        # outputs: (batch_size, block_size, vocab_size)
        tok_emb = self.token_embedding_table(inputs)  # B, T, E
        pos_emb = self.position_embedding_table(torch.arange(inputs.shape[1], device=self.device).unsqueeze(0))  # B, T, E

        x = tok_emb + pos_emb  # B, T, E
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        # B, T, V = logits.shape
        # logits = F.softmax(logits.view(-1, V), dim=-1).reshape(B, T, V)
        return logits

    def sample(self, max_size: int = 128, start_token: int = 0):
        """
        Generates a sequence of tokens using the trained model.

        Args:
            max_size (int, optional): Maximum size of the generated sequence. Default is 128.
            start_token (int, optional): Start token for the generation process. Default is 0.

        Returns:
            torch.Tensor: Generated sequence of tokens.
        """
        tokens = torch.tensor(start_token).reshape(1, -1).to(self.device)

        for _ in range(max_size):
            if len(tokens[0]) >= self.block_size:
                inputs = tokens[:, -self.block_size:]
            else:
                inputs = tokens
            logits = self.forward(inputs)
            logit_next = logits[:, -1, :]
            prob = F.softmax(logit_next, dim=-1)
            token_next = torch.multinomial(prob, num_samples=1)
            tokens = torch.cat([tokens, token_next], dim=1)

        return tokens[0]

    def save(self, path_or_file):
        """
        Saves the model state and hyperparameters to a file.

        Args:
            path_or_file (str or file-like object): Path to save the model or file-like object.
        """
        torch.save(
            {
                "hp": self.hyper_params(),
                "state": self.state_dict(),
            },
            path_or_file
        )



if __name__ == "__main__":
    blm = BigramLangModel(65, block_size=8)
    inputs = torch.randint(0, 65, (4, 8))
    y = blm(inputs)
    sample = y.argmax(dim=-1)
    print(y.shape)
    print(sample)
