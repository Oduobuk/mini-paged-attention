import torch
from typing import Dict, List, Optional


class KVCachePool:
    """
    Global Pre-Allocated Physical Block Memory Pool for PagedAttention.
    
    Instead of reserving continuous GPU tensors per request (causing 60-80% memory waste),
    the pool pre-allocates N fixed physical blocks in HBM/DRAM.
    
    Each physical block stores:
      - Key Cache:   [block_size, num_heads, head_dim]
      - Value Cache: [block_size, num_heads, head_dim]
      
    Features:
      - O(1) allocation and deallocation from a free-list stack.
      - Reference counting for prefix sharing (Copy-on-Write).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu")
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate contiguous physical pool
        self.key_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.value_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )

        # Free-list of available block IDs
        self.free_blocks: List[int] = list(range(num_blocks))
        # Reference count per physical block (for prefix caching / CoW)
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}

    def allocate_block(self) -> int:
        """Allocate a free block from the pool. Raises OOM error if pool is exhausted."""
        if not self.free_blocks:
            raise MemoryError("KVCachePool Out Of Memory: No free physical blocks available.")
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free_block(self, block_id: int) -> None:
        """Decrement reference count; return block to free-list if ref_count hits 0."""
        assert block_id in self.ref_counts, f"Invalid block_id {block_id}"
        if self.ref_counts[block_id] > 0:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.free_blocks.append(block_id)

    def retain_block(self, block_id: int) -> None:
        """Increment reference count (used when sharing prefix blocks across requests)."""
        assert self.ref_counts[block_id] > 0, "Cannot retain an unallocated block"
        self.ref_counts[block_id] += 1

    def copy_on_write(self, block_id: int) -> int:
        """
        If block is shared (ref_count > 1), allocate a new block, copy its contents,
        decrement the old block's ref_count, and return the new private block_id.
        """
        if self.ref_counts[block_id] == 1:
            return block_id  # Already exclusively owned

        new_block_id = self.allocate_block()
        self.key_cache[new_block_id].copy_(self.key_cache[block_id])
        self.value_cache[new_block_id].copy_(self.value_cache[block_id])
        self.free_block(block_id)
        return new_block_id

    def write_kv(
        self,
        block_id: int,
        offset: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """
        Write a single token's Key and Value into a specific slot in a physical block.
        key: [num_heads, head_dim]
        value: [num_heads, head_dim]
        """
        assert 0 <= offset < self.block_size, f"Offset {offset} exceeds block_size {self.block_size}"
        self.key_cache[block_id, offset].copy_(key)
        self.value_cache[block_id, offset].copy_(value)

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def total_memory_bytes(self) -> int:
        element_size = self.key_cache.element_size()
        elements = self.num_blocks * self.block_size * self.num_heads * self.head_dim * 2
        return elements * element_size

