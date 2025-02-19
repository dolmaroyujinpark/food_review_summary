import torch.nn as nn
from .multi_head_attention import SparseMultiHeadAttention
from .feed_forward import FeedForward

class SparseTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, sparsity=5):
        super(SparseTransformerBlock, self).__init__()
        self.attention = SparseMultiHeadAttention(embed_size, heads, sparsity)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        #print("Transformer Block Input Shapes - Value:", value.shape, "Key:", key.shape, "Query:", query.shape)
        attention = self.attention(value, key, query, mask)
        #print("After Attention:", attention.shape)
        x = self.dropout(self.norm1(attention + query))
        #print("After Norm1 and Dropout:", x.shape)
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        #print("Final Block Output Shape:", out.shape)
        return out