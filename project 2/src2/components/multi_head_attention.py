import torch
import torch.nn as nn
import math


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, sparsity=5):
        super(SparseMultiHeadAttention, self).__init__()
        assert embed_size % heads == 0, "Embedding size must be divisible by heads"
        self.heads = heads
        self.head_dim = embed_size // heads
        self.embed_size = embed_size
        self.sparsity = sparsity

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def generate_sparse_mask(self, seq_len, device):
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            mask[i, max(0, i - self.sparsity):min(seq_len, i + self.sparsity + 1)] = True
        return mask

    def forward(self, values, keys, query, mask):
        N, query_len, embed_size = query.shape
        key_len = keys.shape[1]
        head_dim = embed_size // self.heads
        assert embed_size % self.heads == 0, "Embedding size must be divisible by number of heads"

        if key_len != query_len:
            keys = keys[:, :query_len, :]

        sparse_mask = self.generate_sparse_mask(query_len, query.device)

        if mask is not None:
            mask = mask.bool()
            sparse_mask = sparse_mask.bool()
            mask = mask & sparse_mask.unsqueeze(0).unsqueeze(0)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.view(N, query_len, self.heads, head_dim).transpose(1, 2)
        keys = keys.view(N, query_len, self.heads, head_dim).transpose(1, 2)
        queries = queries.view(N, query_len, self.heads, head_dim).transpose(1, 2)

        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / math.sqrt(head_dim), dim=-1)
        out = torch.einsum("nhqk,nhkd->nhqd", [attention, values]).reshape(N, query_len, embed_size)
        out = self.fc_out(out)
        return out