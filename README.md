# Mini-PagedAttention: High-Throughput LLM Memory Architecture from Scratch

[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](https://github.com/Oduobuk/mini-paged-attention)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-PagedAttention%20%7C%20vLLM-purple)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, first-principles implementation of **PagedAttention** and **Continuous Batching** (Kwon et al., vLLM) in Python and PyTorch.

Designed to demystify how production inference engines achieve high throughput under memory constraints, this repository implements **pre-allocated physical block pools, OS-style Page Tables (Block Tables), dynamic token-level iteration scheduling, Copy-on-Write (CoW) prefix sharing, and preemption**, eliminating memory fragmentation and supporting **$3.5\times - 4\times$ higher concurrency** within the exact same VRAM budget.

---

## 🏛️ The Problem: Why Static KV-Cache Allocation Wastes 80%+ of VRAM

In naive LLM serving runtimes (such as standard Hugging Face Transformers or PyTorch implementations), memory for the Key-Value (KV) cache is statically pre-allocated as a single contiguous tensor based on the maximum allowable context length ($S_{\text{max}}$, e.g. 2048 or 4096 tokens):

$$\text{Static Cache Size} = \text{Batch} \times 2 \times \text{Layers} \times \text{Heads} \times S_{\text{max}} \times \text{Head\_Dim} \times \text{Bytes}$$

This causes three catastrophic inefficiencies:
1. **Internal Fragmentation:** Memory reserved for the theoretical maximum sequence length sits completely idle during early token generation (wasting 60–80% of VRAM).
2. **External Fragmentation:** Unpredictable, dynamic output lengths cause physical GPU memory to segment over time, causing Out-Of-Memory (OOM) errors even when nominal free space exists.
3. **Reservation Waste:** Serving systems must cap concurrent batch sizes to tiny limits (e.g. 4–8 requests) to prevent worst-case OOM crashes.

---

## ⚙️ The Solution: Virtual Memory PagedAttention

Inspired by virtual memory paging in operating systems, **PagedAttention** partitions the KV cache into fixed-size physical blocks (e.g., $16$ tokens per block). Non-contiguous physical blocks are mapped dynamically via per-request **Block Tables**:

```
               PAGED ATTENTION VIRTUAL MEMORY ARCHITECTURE
    Logical Token Sequence (Request A):
    [Token 0 ... 15]   [Token 16 ... 31]   [Token 32 ... 47]
           │                   │                   │
           ▼                   ▼                   ▼
     Logical Block 0     Logical Block 1     Logical Block 2
           │                   │                   │
    ┌──────┴───────────────────┴───────────────────┴──────┐
    │          BLOCK TABLE (Page Table per Request)       │
    │  Logical Block 0  ──►  Physical Block 14 (in Pool)  │
    │  Logical Block 1  ──►  Physical Block 3  (in Pool)  │
    │  Logical Block 2  ──►  Physical Block 42 (in Pool)  │
    └──────────────────────────┬──────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────┐
    │        GLOBAL KV CACHE POOL (Pre-Allocated HBM)     │
    │  [Block 0] [Block 1] ... [Block 3] ... [Block 14]   │
    │  (Non-contiguous physical storage, ZERO waste!)     │
    └─────────────────────────────────────────────────────┘
```

---

## 📐 Core Architectural Components

### 1. Global Block Pool (`KVCachePool`)
* Pre-allocates a single contiguous memory tensor of shape `[num_blocks, block_size, num_heads, head_dim]` in HBM.
* Manages an $O(1)$ free-list stack and per-block reference counts (`ref_counts`).

### 2. Request Page Tables (`BlockTable` & `Sequence`)
* Each request tracks its own dynamic block table: `block_table = [p0, p1, ..., pm]`.
* Maps logical token position $t$ to physical block slot:
  $$\text{block\_id} = \text{block\_table}[t // \text{block\_size}], \quad \text{offset} = t \pmod{\text{block\_size}}$$

### 3. Copy-on-Write (CoW) Prefix Sharing
* When multiple requests share a prompt prefix (e.g. system instructions or few-shot examples), they point to the exact same physical blocks in the pool (`ref_count > 1`).
* When a request appends a new private token that alters a shared block, a fresh block is allocated, previous states are copied, and the old block's reference count is decremented.

### 4. Continuous Batching Iteration Scheduler (`ContinuousBatchScheduler`)
* Replaces static batch synchronization with iteration-level scheduling:
  * **Admission:** Incoming requests are admitted as soon as the pool has sufficient blocks for their prompt.
  * **Dynamic Expansion:** Active decode tokens allocate blocks strictly as they cross block boundaries ($t \pmod{16} == 0$).
  * **Immediate Recycling:** The moment a sequence emits `<EOS>` or hits `max_tokens`, its physical blocks are instantly returned to the free pool.
  * **Preemption:** If the pool runs out of physical blocks, lowest-priority sequences are temporarily preempted back to the waiting queue, preventing hard crashes.

---

## 📁 Repository Structure

```
mini-paged-attention/
├── mini_paged_attention/
│   ├── __init__.py                 # Clean package entrypoint
│   ├── block_pool.py               # Pre-allocated physical block pool & CoW manager
│   ├── sequence.py                 # Sequence data structures & Page Tables
│   ├── paged_attention.py          # PagedAttention decode operator & numerical parity
│   └── scheduler.py                # Continuous batching & iteration-level scheduler
├── benchmarks/
│   └── benchmark_fragmentation.py  # Empirical concurrency & fragmentation profiler
├── tests/
│   ├── test_block_pool.py          # Block allocation, OOM guards & CoW verification
│   ├── test_paged_attention.py     # Bit-exact parity tests vs standard attention
│   └── test_scheduler.py           # Lifecycle, preemption & token expansion tests
├── Makefile                        # Top-level test & benchmark commands
├── pyproject.toml                  # Python package configuration
└── README.md                       # Systems architecture documentation
```

---

## 🔬 Empirical Concurrency & Fragmentation Benchmark

Benchmarking a fixed VRAM budget (128 MB) under a realistic request distribution (64 dynamic requests, prompts 32–128 tokens, generations 16–128 tokens):

| Metric | Static Contiguous (Standard HF/PyTorch) | PagedAttention Block Pool (Ours) |
| :--- | :--- | :--- |
| **Fixed VRAM Budget** | 128 MB | 128 MB |
| **Max Concurrent Requests** | 16 requests *(OOM Capped)* | **56 concurrent requests** |
| **Internal Fragmentation** | **83.4% WASTED** *(Reserved for Max Context)* | **< 4.0%** *(Only last block partially filled)* |
| **Throughput Concurrency Gain** | 1.00x *(Baseline)* | **3.50x Higher Concurrency** |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Oduobuk/mini-paged-attention.git
cd mini-paged-attention
pip install -e .
```

### 2. Run Test Suite
```bash
make test
```

### 3. Run Fragmentation & Concurrency Benchmark
```bash
make benchmark
```

### 4. Python API Example
```python
from mini_paged_attention import KVCachePool, Sequence, ContinuousBatchScheduler

# 1. Initialize physical block pool (64 blocks, 16 tokens/block)
pool = KVCachePool(num_blocks=64, block_size=16, num_heads=8, head_dim=64)

# 2. Initialize continuous batch scheduler
scheduler = ContinuousBatchScheduler(block_pool=pool, max_batch_size=16)

# 3. Add incoming requests
seq1 = Sequence(seq_id=1, prompt_token_ids=list(range(32)), max_tokens=64)
scheduler.add_sequence(seq1)

# 4. Step iteration-level continuous generation
while scheduler.has_unfinished_sequences():
    meta = scheduler.step()
    print(f"Running: {meta['num_running']}, Free Blocks: {meta['free_blocks']}")
```

---

## 📄 License
MIT License - see [LICENSE](LICENSE) for details.
