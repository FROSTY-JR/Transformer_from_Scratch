# This is my attempt to build a Transformer from scratch based on the
# "Attention Is All You Need" paper (Vaswani et al., 2017).
# Goal: Understand the math and all the matrix operations, not just use the API.
# I have refined the documentation using gpt to frame sentences better, but the code is not ai generated.

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    This class handles the Positional Encoding.
    The paper uses sin/cos functions instead of learned embeddings.
    This is static, not learned, so it's not a "parameter".
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        'd_model' is the embedding dimension (e.g., 512)
        'max_len' is the longest possible sentence we'll see.
        """
        # Standard PyTorch class setup
        super(PositionalEncoding, self).__init__()
        
        # --- Create the PE matrix ---
        # We need a big matrix of shape (max_len, d_model)
        # It will hold the encoding for every word (up to max_len)
        # at every dimension (up to d_model).
        pe = torch.zeros(max_len, d_model)
        
        # --- Create the 'position' tensor ---
        # This is just a column vector of [0, 1, 2, ..., max_len-1]
        # Shape: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # --- Create the 'div_term' (the denominator) ---
        # This is the 1 / (10000^(2i / d_model)) part.
        # The paper uses 2i and 2i+1. This 'div_term' is for the 2i part.
        # 'torch.arange(0, d_model, 2)' gets us all the even 'i's (0, 2, 4...)
        # The math.log is a trick to calculate 1/10000^x using exp(-log(10000)*x)
        # for numerical stability.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # --- Fill the PE matrix ---
        # Apply the sin/cos functions.
        # Even indices (2i) use sin
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Odd indices (2i + 1) use cos
        # The div_term is the same for both, as per the paper.
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # --- Final prep for the matrix ---
        # The paper adds the PE to the input embeddings.
        # The input 'x' will be (batch_size, seq_len, d_model).
        # We need to add a batch dimension to 'pe' so it can broadcast.
        # Shape becomes: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # --- Register as a 'buffer' ---
        # This is important. 'register_buffer' tells PyTorch that 'pe'
        # is part of the model's state, but it's NOT a parameter that
        # needs to be trained or updated with gradients.
        self.register_buffer('pe', pe)

#Next i will be making a forward class where i will be passing Q, K, V through their linear layers (w_q, w_k, w_v).
