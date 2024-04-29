import torch
import math

def get_positional_encoding(max_len, nhid, device):
    """
    Compute the positional encoding for a given maximum sequence length and hidden size.
    """
    pe = torch.zeros(max_len, nhid, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, nhid, 2, device=device).float() * (-math.log(10000.0) / nhid))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe