from torch import arange
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embedding_dim, n_heads, n_kv_heads, dropout=0.0, is_causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // n_kv_heads
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        self.head_dim = embedding_dim // n_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.query_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_proj = nn.Linear(embedding_dim, n_kv_heads * self.head_dim, bias=False)
        self.value_proj = nn.Linear(embedding_dim, n_kv_heads * self.head_dim, bias=False)

        self.output_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        # We take input with shape (batch_size, context_length, embedding_dim)..
        # We want to project query, key, and value matrices from the input
        # We then reshape and transpose them to have shape (batch_size, n_head, context_length, head_dim)
        # We calculate attention with each of those heads
        # Then we merge the heads back to shape (batch_size, context_length, embedding_dim) and return that

        batch_size, context_length, embedding_dim = x.shape
        query_matrix = self.query_proj(x)
        key_matrix = self.key_proj(x)
        value_matrix = self.value_proj(x)

        query_heads = query_matrix.reshape(batch_size, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        key_heads = key_matrix.reshape(batch_size, context_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value_heads = value_matrix.reshape(batch_size, context_length, self.n_kv_heads, self.head_dim).transpose(1, 2)

        key_heads = key_heads.repeat_interleave(self.n_rep, dim=1)
        value_heads = value_heads.repeat_interleave(self.n_rep, dim=1)

        grouped_query_attention = nn.functional.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            dropout_p = self.dropout if self.training else 0.0,
            is_causal=self.is_causal
        )

        merged_attention = grouped_query_attention.transpose(1, 2).flatten(start_dim=2)
        return self.output_proj(merged_attention)


class FFN(nn.Module):
    def __init__(self, embedding_dim, intermediate_dim):
        super().__init__()

        self.gate_proj = nn.Linear(embedding_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(embedding_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, embedding_dim, bias=False)

    def forward(self, x):
        gated_output = self.gate_proj(x)
        up_output = self.up_proj(x)

        return self.down_proj(F.silu(gated_output) * up_output)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, n_kv_heads, intermediate_dim, dropout=0.0, is_causal=True):
        super().__init__()

        self.norm_1 = nn.RMSNorm(embedding_dim, eps=1e-6)
        self.attention = Attention(embedding_dim, n_heads, n_kv_heads, dropout, is_causal)
        self.norm_2 = nn.RMSNorm(embedding_dim, eps=1e-6)
        self.ffn = FFN(embedding_dim, intermediate_dim)

    def forward(self, x):
        res_x_1 = x
        x = self.norm_1(x)
        x = self.attention(x)
        x = x + res_x_1
        res_x_2 = x
        x = self.norm_2(x)
        x = self.ffn(x)
        return res_x_2 + x


class LLM(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim, n_layers, n_heads, n_kv_heads, intermediate_dim,
                 dropout=0.0, is_causal=True):
        super().__init__()
        self.context_length = context_length

        self.tok_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)

        nn.init.normal_(self.tok_embedding.weight, std=0.02)
        # output_proj shares this weight — std=0.02 keeps initial logits small
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, n_heads, n_kv_heads, intermediate_dim, dropout, is_causal)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.RMSNorm(embedding_dim, eps=1e-6)

        self.output_proj = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_proj.weight = self.tok_embedding.weight

    def forward(self, x):
        batch_size, context_length = x.shape
        pos_embeddings = self.pos_embedding(arange(context_length, device=x.device))
        x = self.tok_embedding(x) + pos_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.output_proj(x)