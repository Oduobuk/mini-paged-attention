from enum import Enum
from typing import List, Optional


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    FINISHED = "FINISHED"


class Sequence:
    """
    Represents an ongoing LLM generation request with dynamic sequence length
    and an OS-like Page Table (Block Table) mapping logical blocks to physical blocks.
    """

    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: List[int],
        max_tokens: int = 64,
        block_size: int = 16
    ):
        self.seq_id = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.max_tokens = max_tokens
        self.block_size = block_size
        self.status = SequenceStatus.WAITING

        # Block Table: maps logical block index i to physical_block_id in KVCachePool
        self.block_table: List[int] = []

    def get_len(self) -> int:
        """Total current sequence length (prompt + generated tokens)."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def num_logical_blocks(self) -> int:
        """Number of logical blocks required to hold the current sequence."""
        seq_len = self.get_len()
        if seq_len == 0:
            return 0
        return (seq_len + self.block_size - 1) // self.block_size

    def last_block_offset(self) -> int:
        """Slot index (0 to block_size - 1) of the latest token within its physical block."""
        seq_len = self.get_len()
        if seq_len == 0:
            return 0
        return (seq_len - 1) % self.block_size

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED or len(self.output_token_ids) >= self.max_tokens

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if len(self.output_token_ids) >= self.max_tokens:
            self.status = SequenceStatus.FINISHED

