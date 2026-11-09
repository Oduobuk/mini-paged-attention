import math
import torch
from typing import List


def paged_attention_forward(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: List[List[int]],
    seq_lens: List[int],
    scale: float,
    block_size: int = 16
) -> torch.Tensor:
    """
    PagedAttention Decode Operator:
    Computes attention between Query tokens and Key/Value states scattered across
    non-contiguous physical blocks in the memory pool.

    Args:
        query: [batch_size, num_heads, head_dim] Query vectors for current decode token.
        key_cache: [num_blocks, block_size, num_heads, head_dim] Global pool.
        value_cache: [num_blocks, block_size, num_heads, head_dim] Global pool.
        block_tables: List of physical block IDs for each sequence in batch.
        seq_lens: Current sequence length of each sequence in batch.
        scale: Attention scaling factor (typically 1.0 / sqrt(head_dim)).
        block_size: Number of tokens per physical block.

    Returns:
        output: [batch_size, num_heads, head_dim] Attention output vectors.
    """
    batch_size, num_heads, head_dim = query.shape
    device = query.device
    output = torch.zeros_like(query)

    for i in range(batch_size):
        seq_len = seq_lens[i]
        block_table = block_tables[i]
        q_i = query[i]  # [num_heads, head_dim]

        # Gather Keys and Values from non-contiguous physical blocks
        k_blocks = []
        v_blocks = []

        tokens_remaining = seq_len
        for block_id in block_table:
            tokens_in_block = min(tokens_remaining, block_size)
            k_blocks.append(key_cache[block_id, :tokens_in_block])    # [tokens_in_block, num_heads, head_dim]
            v_blocks.append(value_cache[block_id, :tokens_in_block])  # [tokens_in_block, num_heads, head_dim]
            tokens_remaining -= tokens_in_block
            if tokens_remaining <= 0:
                break

        k_all = torch.cat(k_blocks, dim=0)  # [seq_len, num_heads, head_dim]
        v_all = torch.cat(v_blocks, dim=0)  # [seq_len, num_heads, head_dim]

        # Reshape for multi-head attention:
        # q: [num_heads, 1, head_dim]
        # k: [num_heads, head_dim, seq_len]
        # v: [num_heads, seq_len, head_dim]
        q_heads = q_i.unsqueeze(1)                          # [num_heads, 1, head_dim]
        k_heads = k_all.permute(1, 2, 0)                    # [num_heads, head_dim, seq_len]
        v_heads = v_all.permute(1, 0, 2)                    # [num_heads, seq_len, head_dim]

        # Scaled dot-product: Q @ K^T / sqrt(d)
        scores = torch.bmm(q_heads, k_heads) * scale        # [num_heads, 1, seq_len]

        # Softmax over sequence dimension
        attn_weights = torch.softmax(scores, dim=-1)        # [num_heads, 1, seq_len]

        # Context output: Attn @ V
        out_i = torch.bmm(attn_weights, v_heads)            # [num_heads, 1, head_dim]
        output[i] = out_i.squeeze(1)

    return output


def paged_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    Standard contiguous PyTorch attention baseline for numerical verification.
    query: [batch_size, num_heads, head_dim]
    keys:   [batch_size, seq_len, num_heads, head_dim]
    values: [batch_size, seq_len, num_heads, head_dim]
    """
    q = query.unsqueeze(2).permute(0, 1, 2, 3)         # [batch, heads, 1, head_dim]
    k = keys.permute(0, 2, 3, 1)                       # [batch, heads, head_dim, seq_len]
    v = values.permute(0, 2, 1, 3)                     # [batch, heads, seq_len, head_dim]

    scores = torch.matmul(q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)                       # [batch, heads, 1, head_dim]
    return out.squeeze(2)
