import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        self.encoding = torch.zeros(max_len, embed_size)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))

        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        #print("Positional Encoding Input Shape:", x.shape)
        if len(x.size()) != 3:
            raise ValueError(f"Expected input x to have 3 dimensions, but got {len(x.size())} dimensions")
        if x.size(2) != self.embed_size:
            raise ValueError(f"Input embedding size {x.size(2)} does not match positional encoding size {self.embed_size}")
        out = x + self.encoding[:, :x.size(1), :].to(x.device)
        #print("After Adding Positional Encoding:", out.shape)
        return out