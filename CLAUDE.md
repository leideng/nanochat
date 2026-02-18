# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a minimal, end-to-end LLM training and deployment system by Andrej Karpathy. It trains and runs ChatGPT-like models on a single GPU node (~3 hours on 8xH100 for GPT-2 grade). Python 3.10+, PyTorch 2.9.1, managed with `uv`.

## Commands

### Setup
```bash
uv venv && uv sync --extra gpu && source .venv/bin/activate  # GPU
uv venv && uv sync --extra cpu && source .venv/bin/activate  # CPU
```

### Tests
```bash
pytest tests/ -v                    # all tests
pytest tests/test_engine.py -v      # specific file
pytest -m "not slow"                # skip slow tests
```

### Training pipeline (8xH100)
```bash
bash runs/speedrun.sh                                        # full pipeline end-to-end
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --fp8
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

### Inference
```bash
python -m scripts.chat_cli -p "Why is the sky blue?"   # CLI (single prompt)
python -m scripts.chat_cli                              # CLI (interactive)
python -m scripts.chat_web                              # Web UI (FastAPI)
```

### Utilities
```bash
python -m nanochat.dataset -n 370        # download data shards
python -m scripts.tok_train              # train tokenizer
python -m scripts.tok_eval               # evaluate tokenizer
python -m nanochat.report reset          # reset training report
python -m nanochat.report generate       # generate report
```

## Architecture

**4-stage pipeline:** Tokenization → Pretraining → SFT → Inference

### Core modules (`nanochat/`)
- **gpt.py** — GPT-2 style Transformer. `--depth` is the single complexity dial that auto-computes all other hyperparams (width, heads, GQA groups, sequence length, batch size). Uses RoPE, QK-norm, ReLU² MLP, value embeddings with gating, sliding window attention.
- **engine.py** — Inference engine with KV cache (FA3-compatible layout). Handles temperature/top-k sampling, streaming, multi-sample generation.
- **optim.py** — Hybrid optimizer: Muon (orthogonalization-based) for weight matrices, AdamW for embeddings/scalars. Both single-GPU (`MuonAdamW`) and distributed (`DistMuonAdamW`) versions.
- **dataloader.py** — Distributed data loader with BOS-aligned best-fit packing. Every row starts with BOS; documents packed to minimize cropping.
- **fp8.py** — Minimal FP8 training (~150 lines, replaces torchao). Uses `torch._scaled_mm` with e4m3fn for fwd, e5m2 for gradients.
- **tokenizer.py** — BPE tokenizer with HuggingFace (training) and tiktoken/RustBPE (inference) backends. Vocab size 32768.
- **checkpoint_manager.py** — Save/load model + optimizer state, handles DDP synchronization.

### Scripts (`scripts/`)
Entry points run as `python -m scripts.<name>`. Training scripts use `torchrun` for multi-GPU.

### Tasks (`tasks/`)
Evaluation task families: ARC, MMLU, GSM8K, HumanEval, SpellingBee, SmolTalk, CustomJSON. Base classes in `tasks/common.py` (TaskMixture, TaskSequence).

## Key Design Decisions

- **Single `--depth` parameter** controls model size; all other hyperparams derived automatically via scaling laws (Chinchilla-optimal data:param ratio of 10.5).
- **No torchao dependency** — custom FP8 implementation is simpler and ~3% faster.
- **FlashAttention-3** on Hopper+ GPUs, falls back to PyTorch SDPA.
- **`print0()`** for rank-0-only logging in distributed training.
- **`torch.compile()`** used for optimizer kernels and model components.

## Environment Variables

- `NANOCHAT_BASE_DIR` — cache/artifacts directory (default: `~/.cache/nanochat`)
- `WANDB_RUN` — wandb run name (empty or "dummy" disables logging)
- `OMP_NUM_THREADS=1` — set during training
