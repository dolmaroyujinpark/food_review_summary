import torch
import torch.nn as nn
from components.transformer_block import SparseTransformerBlock
from components.positional_encoding import PositionalEncoding



class SparseEncoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, sparsity=5):
        super(SparseEncoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList(
            [SparseTransformerBlock(embed_size, heads, dropout, forward_expansion, sparsity) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        #print("Encoder Input Shape:", x.shape)
        x = self.word_embedding(x)
        #print("After Word Embedding:", x.shape)
        x = x + self.position_embedding(x)
        #print("After Positional Encoding:", x.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x, x, x, mask)
            #print(f"After Encoder Layer {i + 1}:", x.shape)
        return x


class SparseDecoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length, sparsity=5):
        super(SparseDecoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList(
            [SparseTransformerBlock(embed_size, heads, dropout, forward_expansion, sparsity) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        #print("Decoder Input Shape:", x.shape)
        x = self.word_embedding(x)
        #print("After Word Embedding:", x.shape)
        x = x + self.position_embedding(x)
        #print("After Positional Encoding:", x.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x, enc_out, x, trg_mask)
            #print(f"After Decoder Layer {i + 1}:", x.shape)
        x = self.fc_out(x)
        #print("Final Decoder Output Shape:", x.shape)
        return x


class SparseTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=3,
        forward_expansion=4,
        heads=8,
        dropout=0.3,
        device="cuda",
        max_length=5000,
        sparsity=5,
    ):
        super(SparseTransformer, self).__init__()
        self.encoder = SparseEncoder(
            src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, sparsity
        )
        self.decoder = SparseDecoder(
            trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length, sparsity
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #print("Source Mask Shape:", src_mask.shape)
        return src_mask.to(src.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        #print("Target Mask Shape:", trg_mask.shape)
        return trg_mask.to(trg.device)

    def forward(self, src, trg):
        #print("Transformer Input Shapes:", f"src1: {src1.shape}, trg: {trg.shape}")
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        #print("Encoder Output Shape:", enc_src.shape)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        #print("Transformer Output Shape:", out.shape)
        return out