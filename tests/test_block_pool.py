import pytest
import torch
from mini_paged_attention.block_pool import KVCachePool


def test_block_allocation_and_free():
    pool = KVCachePool(num_blocks=4, block_size=16, num_heads=4, head_dim=32)
    assert pool.num_free_blocks() == 4

    b0 = pool.allocate_block()
    b1 = pool.allocate_block()
    assert pool.num_free_blocks() == 2
    assert pool.ref_counts[b0] == 1
    assert pool.ref_counts[b1] == 1

    pool.free_block(b0)
    assert pool.num_free_blocks() == 3
    assert pool.ref_counts[b0] == 0

    pool.free_block(b1)
    assert pool.num_free_blocks() == 4


def test_pool_exhaustion_raises_oom():
    pool = KVCachePool(num_blocks=2, block_size=16, num_heads=2, head_dim=16)
    _ = pool.allocate_block()
    _ = pool.allocate_block()
    assert pool.num_free_blocks() == 0

    with pytest.raises(MemoryError, match="Out Of Memory"):
        pool.allocate_block()


def test_copy_on_write():
    pool = KVCachePool(num_blocks=4, block_size=4, num_heads=2, head_dim=8)
    b0 = pool.allocate_block()

    # Write data to b0
    k_data = torch.randn(2, 8)
    v_data = torch.randn(2, 8)
    pool.write_kv(b0, offset=0, key=k_data, value=v_data)

    # Sequence 2 shares b0 (prefix caching)
    pool.retain_block(b0)
    assert pool.ref_counts[b0] == 2

    # Sequence 2 triggers CoW
    new_b = pool.copy_on_write(b0)
    assert new_b != b0
    assert pool.ref_counts[b0] == 1
    assert pool.ref_counts[new_b] == 1

    # Verify copied data is identical
    assert torch.equal(pool.key_cache[b0, 0], pool.key_cache[new_b, 0])
    assert torch.equal(pool.value_cache[b0, 0], pool.value_cache[new_b, 0])
