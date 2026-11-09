import pytest
from mini_paged_attention.block_pool import KVCachePool
from mini_paged_attention.sequence import Sequence, SequenceStatus
from mini_paged_attention.scheduler import ContinuousBatchScheduler


def test_scheduler_lifecycle():
    # Pool with 8 blocks (16 tokens per block)
    pool = KVCachePool(num_blocks=8, block_size=16)
    scheduler = ContinuousBatchScheduler(block_pool=pool, max_batch_size=4)

    # Add 2 sequences with short generations (prompt=10, max=5 tokens)
    s1 = Sequence(seq_id=1, prompt_token_ids=list(range(10)), max_tokens=3, block_size=16)
    s2 = Sequence(seq_id=2, prompt_token_ids=list(range(8)), max_tokens=2, block_size=16)

    scheduler.add_sequence(s1)
    scheduler.add_sequence(s2)

    assert pool.num_free_blocks() == 8

    # Step 1: Admission
    step1 = scheduler.step()
    assert step1["num_running"] == 2
    assert step1["num_waiting"] == 0
    # Both sequences took 1 block each
    assert pool.num_free_blocks() == 6

    # Step 2: Generation
    step2 = scheduler.step()
    # s2 finishes (2 tokens generated)
    assert 2 in step2["finished_ids"]
    assert pool.num_free_blocks() == 7  # s2's block freed!

    # Step 3: s1 finishes (3 tokens generated)
    step3 = scheduler.step()
    assert 1 in step3["finished_ids"]
    assert pool.num_free_blocks() == 8  # All blocks completely recycled!
    assert not scheduler.has_unfinished_sequences()


def test_scheduler_preemption_on_memory_pressure():
    # Only 2 blocks available
    pool = KVCachePool(num_blocks=2, block_size=4)
    scheduler = ContinuousBatchScheduler(block_pool=pool, max_batch_size=2)

    # s1 prompt=4 tokens (uses 1 block)
    s1 = Sequence(seq_id=1, prompt_token_ids=[1, 2, 3, 4], max_tokens=10, block_size=4)
    # s2 prompt=4 tokens (uses 1 block)
    s2 = Sequence(seq_id=2, prompt_token_ids=[5, 6, 7, 8], max_tokens=10, block_size=4)

    scheduler.add_sequence(s1)
    scheduler.add_sequence(s2)

    # Step 1: Both admitted (pool has 0 free blocks)
    scheduler.step()
    assert pool.num_free_blocks() == 0

    # Step 2: Next token pushes length to 5, which requires a 2nd block!
    # Because pool has 0 blocks, scheduler must preempt a sequence
    step2 = scheduler.step()
    assert len(step2["preempted_ids"]) > 0
    assert step2["num_running"] < 2
