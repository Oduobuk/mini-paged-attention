from typing import List, Dict, Any, Optional
from collections import deque
from .block_pool import KVCachePool
from .sequence import Sequence, SequenceStatus


class ContinuousBatchScheduler:
    """
    Iteration-Level Continuous Batching Scheduler.
    
    Dynamically schedules sequences token-by-token:
    1. Admits pending prefill requests when physical block capacity allows.
    2. Dynamically expands KV blocks for active decode tokens.
    3. Triggers preemption (swapping/re-queueing) if physical blocks are exhausted.
    4. Immediately evicts physical blocks back to the pool upon sequence completion.
    """

    def __init__(
        self,
        block_pool: KVCachePool,
        max_batch_size: int = 16
    ):
        self.block_pool = block_pool
        self.max_batch_size = max_batch_size

        self.waiting_queue: deque[Sequence] = deque()
        self.running_queue: List[Sequence] = []
        self.finished_queue: List[Sequence] = []

    def add_sequence(self, seq: Sequence) -> None:
        """Enqueue a new incoming inference request."""
        seq.status = SequenceStatus.WAITING
        self.waiting_queue.append(seq)

    def step(self) -> Dict[str, Any]:
        """
        Execute a single iteration step:
        - Admit waiting sequences if blocks are available.
        - Check memory allocation for active decode steps.
        - Preempt sequences if physical blocks run out.
        - Advance token generation and free completed sequences.
        """
        preempted_ids = []

        # Step 1: Check running sequences for needed block allocations
        # A sequence needs a new physical block when its logical block requirement exceeds allocated blocks
        for seq in list(self.running_queue):
            if len(seq.block_table) < seq.num_logical_blocks():
                if self.block_pool.num_free_blocks() > 0:
                    new_block = self.block_pool.allocate_block()
                    seq.block_table.append(new_block)
                else:
                    # OOM: Preempt this sequence (LIFO preemption)
                    self._preempt_sequence(seq)
                    preempted_ids.append(seq.seq_id)

        # Step 2: Admit waiting sequences if batch slots & blocks are available
        while self.waiting_queue and len(self.running_queue) < self.max_batch_size:
            candidate = self.waiting_queue[0]
            blocks_needed = candidate.num_logical_blocks()

            if self.block_pool.num_free_blocks() >= blocks_needed:
                self.waiting_queue.popleft()
                for _ in range(blocks_needed):
                    candidate.block_table.append(self.block_pool.allocate_block())
                candidate.status = SequenceStatus.RUNNING
                self.running_queue.append(candidate)
            else:
                # Cannot admit more sequences in this iteration
                break

        # Step 3: Advance active sequences by 1 token step
        finished_ids = []
        for seq in list(self.running_queue):
            # Simulate generating a token
            dummy_next_token = 1000 + seq.get_len()
            seq.append_token(dummy_next_token)

            if seq.is_finished():
                self._finish_sequence(seq)
                finished_ids.append(seq.seq_id)

        # Compute telemetry
        total_blocks = self.block_pool.num_blocks
        used_blocks = total_blocks - self.block_pool.num_free_blocks()
        utilization = (used_blocks / total_blocks) * 100.0 if total_blocks > 0 else 0.0

        return {
            "num_running": len(self.running_queue),
            "num_waiting": len(self.waiting_queue),
            "num_finished": len(self.finished_queue),
            "finished_ids": finished_ids,
            "preempted_ids": preempted_ids,
            "free_blocks": self.block_pool.num_free_blocks(),
            "utilization_pct": utilization
        }

    def _finish_sequence(self, seq: Sequence) -> None:
        """Free all physical blocks associated with a completed sequence."""
        for block_id in seq.block_table:
            self.block_pool.free_block(block_id)
        seq.block_table.clear()
        seq.status = SequenceStatus.FINISHED
        self.running_queue.remove(seq)
        self.finished_queue.append(seq)

    def _preempt_sequence(self, seq: Sequence) -> None:
        """Preempt a running sequence, free its physical blocks, and return to waiting queue."""
        for block_id in seq.block_table:
            self.block_pool.free_block(block_id)
        seq.block_table.clear()
        seq.status = SequenceStatus.PREEMPTED
        self.running_queue.remove(seq)
        self.waiting_queue.appendleft(seq)

    def has_unfinished_sequences(self) -> bool:
        return len(self.waiting_queue) > 0 or len(self.running_queue) > 0
