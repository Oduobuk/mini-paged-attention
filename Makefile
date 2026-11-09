PYTHON ?= python3

.PHONY: help test benchmark clean

help:
	@echo "Mini-PagedAttention - Command Menu:"
	@echo "  make test       - Run all unit and parity tests (BlockPool, PagedAttention, Scheduler)"
	@echo "  make benchmark  - Run memory fragmentation & concurrency benchmark"
	@echo "  make clean      - Remove build artifacts and caches"

test:
	$(PYTHON) -m pytest tests/ -v

benchmark:
	$(PYTHON) benchmarks/benchmark_fragmentation.py

clean:
	rm -rf build *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

