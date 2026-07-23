# 2,000-Step LLM Speedrun — Bilingual EN+HI

> Minimize **bits-per-byte (BPB)** on mixed English + Hindi text  
> Constraints: ≤2,000 optimizer steps · ≤2M parameters · CPU only · pure PyTorch

## The Challenge

The starter kit ships a deliberately mediocre GPT: byte tokenizer (vocab=256), fixed LR, batch=8, no warmup, no weight tying. Dev BPB: **2.3718**.

Every improvement must come from better use of the *same* fixed resources — no GPU, no pretrained weights, no external data.

## Key Insight: The Hindi Byte Problem

Analyzing the corpus revealed that **14% of characters (Devanagari/Hindi) consume 33% of bytes** under UTF-8, because each Hindi character encodes to 3 bytes. The byte tokenizer therefore wastes one-third of the context window on character-internal byte structure with no semantic signal.

```
block_size=128 (byte tokenizer): ~128 English chars OR ~42 Hindi chars of context
block_size=384 (BPE-4096):       ~384 English chars OR ~384 Hindi chars of context
```

BPE with Devanagari-aware pre-tokenization collapses this waste.

## Results

| Run | Key Change | Dev BPB | Δ vs Baseline |
|-----|-----------|---------|--------------|
| 0 | Baseline (as given) | 2.3718 | — |
| 1 | Batch 32, cosine LR, grad clip, weight tying | ~2.05 | −13% |
| 2 | BPE-2048 tokenizer | ~1.75 | −26% |
| 3 | BPE-4096 (sweet spot in vocab sweep) | ~1.70 | −28% |
| 4 | RMSNorm + RoPE + GPT-2 init | ~1.67 | −30% |
| 5 | Muon optimizer (2D matrix weights) | ~1.64 | −31% |
| 6 | block_size 128→384 (works under Muon) | ~1.61 | −32% |
| 7 | WSD schedule + EMA averaging | **~1.59** | **−33%** |

## Architecture

```
n_embd    = 176
n_layer   = 3
n_head    = 8
block_size= 384
vocab     = 4096 (BPE with Devanagari-aware merges + byte fallback)
pos_enc   = RoPE
norm      = RMSNorm
weight_tying = True (functional, one copy in checkpoint)
params    = ~1.84M
```

## Reproduce

```bash
# 1. Train tokenizer (~2 min)
python train_tokenizer.py --data data/train_corpus.txt --merges 3840

# 2. Train model (~3-4 min on CPU)
python train.py \
  --data data/train_corpus.txt --steps 2000 --out ckpt.pt \
  --optimizer muon --muon_lr 0.02 --lr 3e-3 \
  --n_embd 176 --n_layer 3 --n_head 8 --block_size 384 \
  --batch 32 --schedule wsd --warmup 150 --decay_start 1600 \
  --ema --ema_decay 0.995 --ema_start 1500

# 3. Evaluate
python evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
```

## Files

| File | Purpose |
|------|---------|
| `model.py` | GPT with RMSNorm, RoPE, functional weight tying |
| `tokenizer.py` | BPE tokenizer, Devanagari-aware, lossless byte fallback |
| `train_tokenizer.py` | Trains and saves `tokenizer.json` |
| `train.py` | Training loop (Muon + WSD + EMA) |
| `optim.py` | Muon optimizer with Newton-Schulz orthogonalization |
| `evaluate.py` | Official BPB scorer (unmodified interface) |
| `RUNLOG.md` | Every experiment: hypothesis → result → conclusion |
| `NOTES.md` | Final config analysis and what did/didn't work |
| `tokenizer.json` | Trained BPE vocabulary |
| `ckpt.pt` | Best checkpoint |
