import pytest
import torch
import math
from mini_paged_attention.block_pool import KVCachePool
from mini_paged_attention.paged_attention import paged_attention_forward, paged_attention_reference


def test_paged_attention_parity_vs_reference():
    """
    Test mathematical equivalence between PagedAttention (scattered physical blocks)
    and unpaged contiguous standard PyTorch attention.
    """
    torch.manual_seed(42)
    batch_size = 2
    num_heads = 4
    head_dim = 32
    block_size = 16
    scale = 1.0 / math.sqrt(head_dim)

    # Sequence lengths: seq 0 has 20 tokens, seq 1 has 35 tokens
    seq_lens = [20, 35]

    pool = KVCachePool(num_blocks=16, block_size=block_size, num_heads=num_heads, head_dim=head_dim)

    # Allocate blocks non-contiguously in scrambled physical order
    # Seq 0: needs 2 blocks -> e.g. physical block 7, 2
    # Seq 1: needs 3 blocks -> e.g. physical block 11, 0, 5
    block_tables = [
        [7, 2],
        [11, 0, 5]
    ]

    # Ground-truth continuous K and V tensors
    k_ref = torch.zeros(batch_size, max(seq_lens), num_heads, head_dim)
    v_ref = torch.zeros(batch_size, max(seq_lens), num_heads, head_dim)

    # Populate pool and ref tensors with identical values
    for i in range(batch_size):
        length = seq_lens[i]
        tokens_remaining = length
        tok_idx = 0

        for b_id in block_tables[i]:
            chunk = min(tokens_remaining, block_size)
            for offset in range(chunk):
                k_val = torch.randn(num_heads, head_dim)
                v_val = torch.randn(num_heads, head_dim)

                pool.write_kv(b_id, offset, k_val, v_val)
                k_ref[i, tok_idx] = k_val
                v_ref[i, tok_idx] = v_val
                tok_idx += 1

            tokens_remaining -= chunk

    # Query for current decode token: [batch_size, num_heads, head_dim]
    query = torch.randn(batch_size, num_heads, head_dim)

    # 1. Run PagedAttention
    out_paged = paged_attention_forward(
        query=query,
        key_cache=pool.key_cache,
        value_cache=pool.value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        block_size=block_size
    )

    # 2. Run Reference Attention for each sequence
    out_ref_0 = paged_attention_reference(query[0:1], k_ref[0:1, :seq_lens[0]], v_ref[0:1, :seq_lens[0]], scale)
    out_ref_1 = paged_attention_reference(query[1:2], k_ref[1:2, :seq_lens[1]], v_ref[1:2, :seq_lens[1]], scale)
    out_ref = torch.cat([out_ref_0, out_ref_1], dim=0)

    # Verify bit-exact numerical parity (max error < 1e-6)
    max_diff = torch.max(torch.abs(out_paged - out_ref)).item()
    assert max_diff < 1e-6, f"PagedAttention parity failed: max diff = {max_diff}"


@pytest.mark.parametrize("head_dim", [16, 32, 64])
@pytest.mark.parametrize("block_size", [8, 16])
def test_paged_attention_varying_dimensions(head_dim, block_size):
    num_heads = 2
    scale = 1.0 / math.sqrt(head_dim)
    pool = KVCachePool(num_blocks=8, block_size=block_size, num_heads=num_heads, head_dim=head_dim)

    b0 = pool.allocate_block()
    b1 = pool.allocate_block()
    block_tables = [[b0, b1]]
    seq_lens = [block_size + 3]

    query = torch.randn(1, num_heads, head_dim)
    out = paged_attention_forward(
        query, pool.key_cache, pool.value_cache, block_tables, seq_lens, scale, block_size=block_size
    )
    assert out.shape == (1, num_heads, head_dim)
    assert not torch.isnan(out).any()
