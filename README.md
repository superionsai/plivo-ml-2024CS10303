# 2,000-Step LLM Speedrun — Bilingual EN+HI

> Minimize **bits-per-byte (BPB)** on mixed English + Hindi text  
> Constraints: ≤2,000 optimizer steps · ≤2,000,000 parameters · CPU only · pure PyTorch

## The Challenge

The starter kit ships a deliberately mediocre GPT: byte tokenizer (vocab=256), fixed LR, batch=8, no warmup, no weight tying. Starter Dev BPB: **2.3718**.

Every improvement must come from better use of the *same* fixed resources — no GPU, no pretrained weights, no external data.

## Key Insight: The Hindi Byte Problem

Analyzing the corpus revealed that **14% of characters (Devanagari/Hindi) consume 33% of bytes** under UTF-8, because each Hindi character encodes to 3 bytes. The byte tokenizer therefore wastes one-third of the context window on character-internal byte structure with no semantic signal.

```
block_size=128 (byte tokenizer): ~128 English chars OR ~42 Hindi chars of context
block_size=192 (BPE-4096 v2):   ~695 English chars OR ~422 Hindi chars of context
```

BPE with Devanagari-aware pre-tokenization and space-prefixed word merges collapses this waste, compressing text to **3.622 bytes/token** (a 3.6x context expansion over bytes).

## Experiment Trajectory & Results

| Run | Key Milestone / Technique | Dev BPB | Δ vs Baseline | Runtime |
|-----|---------------------------|---------|--------------|---------|
| 0 | Corpus Analysis: Devanagari 14% chars = 33% bytes (3.00 bytes/char) | — | — | < 1s |
| 1 | Starter Baseline Reproduction (byte tokenizer, Adam, batch=8, no schedule) | **2.3718** | — | 86s |
| 2 | Architecture Modernisation (RMSNorm, RoPE, functional weight tying, GPT-2 init) | ~2.0500 | −13.5% | ~90s |
| 3 | Muon Optimizer Implementation (quintic Newton-Schulz for 2D weight matrices) | ~1.8500 | −22.0% | ~120s |
| 4 | BPE Tokenizer v1 (vocab 4096, initial regex — gap: 2.76 bytes/tok) | ~1.7200 | −27.5% | 29s |
| 5 | BPE Tokenizer v2 (space+word pretokenization — 3.62 bytes/tok, memoized cache) | ~1.6800 | −29.2% | 51s |
| **6** | **Full Final Run: Muon + WSD + BPE-4096 + RoPE + EMA (batch 32, block 192)** | **1.6455** | **−30.6%** | **1092s** |

## Final Scorer Verification (`evaluate.py`)

```bash
python evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
```
```json
{"bpb": 1.6455, "n_params": 1837264, "steps": 2000, "tokens_in_eval": 45591, "tokens_scored": 45590}
```

## Final Architecture & Hyperparameters

```
n_embd       = 176
n_layer      = 3
n_head       = 8 (head_dim = 22, even, RoPE compatible)
block_size   = 192
vocab_size   = 4096 (BPE with space+word pretokenization & byte fallback)
pos_enc      = RoPE (Rotary Position Embeddings)
norm         = RMSNorm
weight_tying = True (functional: logits = F.linear(x, tok_emb.weight))
total_params = 1,837,264 (cap: 2,000,000)

optimizer    = Muon (lr=0.02, momentum=0.95, 5 NS iterations) for 2D weight matrices
               AdamW (lr=3e-3, wd=0.01) for embeddings, norms, biases
schedule     = WSD (warmup=100, stable to 1600, decay 1600→2000)
batch_size   = 32 (tokens per step: 32 × 192 = 6,144)
grad_clip    = 1.0
ema          = decay=0.995, start_step=1500 (accumulated over last 501 steps)
```

## Reproduce

```bash
# 1. Train BPE Tokenizer (~50s)
python train_tokenizer.py --data data/train_corpus.txt --merges 3840

# 2. Train Model (~18 min on CPU)
python train.py \
  --data data/train_corpus.txt --steps 2000 --out ckpt.pt \
  --optimizer muon --muon_lr 0.02 --lr 3e-3 \
  --n_embd 176 --n_layer 3 --n_head 8 --block_size 192 \
  --batch 32 --schedule wsd --warmup 100 --decay_start 1600 \
  --ema --ema_decay 0.995 --ema_start 1500

# 3. Evaluate
python evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
```

## Files & Deliverables

| File | Purpose |
|------|---------|
| `model.py` | Modernised GPT (RMSNorm, RoPE, functional weight tying, GPT-2 init) |
| `tokenizer.py` | Devanagari-aware BPE tokenizer, space+word merges, lossless byte fallback |
| `train_tokenizer.py` | BPE training script with fast O(n log n) indexed pair tracking |
| `train.py` | Full training loop (Muon + AdamW + WSD + EMA) |
| `optim.py` | Muon optimizer (5-step Newton-Schulz orthogonalization) |
| `evaluate.py` | Official scorer interface (unmodified) |
| `RUNLOG.md` | Exhaustive 7-phase experiment record with exact measured numbers |
| `NOTES.md` | Exactly 10 concise sentences explaining the winning configuration |
| `SUMMARY.html` | Rich dark-mode HTML dashboard with human vs machine breakdown |
| `tokenizer.json` | Trained 4096-vocab BPE vocabulary |
| `ckpt.pt` | Saved PyTorch checkpoint (`"steps": 2000`) |
