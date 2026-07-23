# NOTES — Configuration Analysis and Key Findings

## Final Model Configuration

*(updated after all experiments)*

| Parameter | Value | Why |
|-----------|-------|-----|
| Tokenizer | BPE vocab=4096 | Best compression:context tradeoff on this corpus |
| n_embd | 176 | Wider+shallower wins; frees params via weight tying |
| n_layer | 3 | Shallower = better gradient flow on small data |
| n_head | 8 | head_dim=22 (176/8) — small but sufficient |
| block_size | 384 | Longer context pays off under Muon's fast convergence |
| pos_enc | rope | RoPE: saves ~(384*176)=67k params vs learned, better length gen |
| normalization | RMSNorm | Fewer params than LayerNorm, same quality empirically |
| weight_tying | True | Output logits reuse tok_emb.weight — saves ~vocab*embd params |
| optimizer | Muon+AdamW | Muon for 2D matmuls, AdamW for emb/norm/bias |
| muon_lr | 0.02 | Optimal from sweep |
| adamw_lr | 3e-3 | Standard for small models |
| schedule | WSD | warmup=150, stable to 1600, cosine decay to 2000 |
| batch_size | 32 | 4x baseline (less gradient noise per step) |
| ema | True | EMA from step 1500, decay=0.995 |
| init | GPT-2 style | std=0.02, residual proj scaled by 1/sqrt(2*n_layer) |

## What Moved the Needle (in order of impact)

1. **BPE Tokenizer** — Single biggest win (~0.60 BPB). Devanagari went from
   3 tokens/char (byte) to ~1 token/char. Each training step now sees 3x more
   real semantic context for the Hindi portion of the corpus.

2. **Muon Optimizer** — ~0.024 BPB over AdamW with equal steps. Orthogonalising
   the momentum matrix equalises singular value updates, making 2000 steps go
   much further on the hidden 2D weight matrices.

3. **Longer context (block_size 384)** — Inert under AdamW (model can't converge
   fast enough to use extra context in 2000 steps). Under Muon: clearly beneficial.

4. **RoPE + RMSNorm** — ~0.026 BPB. Saves ~45k params via no learned position
   table, frees space for slightly wider embeddings.

5. **EMA averaging** — ~0.01 BPB. Weight average over final 500 steps gives
   smoother parameter estimate with no additional compute.

6. **WSD schedule vs cosine** — ~0.005-0.01 BPB. Holding LR steady longer before
   decaying gives more useful mid-training signal.

## What Did NOT Help

- **SwiGLU**: param-matched SwiGLU gave no gain over GELU on this task/scale.
  Extra parameters went to narrower MLP which cancelled the activation benefit.
- **BPE-8192**: Diminishing returns — rarer tokens get fewer gradient updates in
  2000 steps. The per-token prediction gets harder while compression gain narrows.
- **More layers (L=5)**: Deeper model underfit more — gradient signal dissipated
  across too many residual connections for this step budget.
- **Dropout > 0**: No improvement; dataset is large enough relative to model size
  that regularisation is not needed.

## Corpus Analysis

```
Total bytes:          7,318,592
Total chars:          5,703,936
Devanagari chars:       801,846  (14.1%)
Devanagari bytes:     2,405,538  (32.9% of total bytes)
Bytes/Devanagari char:     3.00  <- UTF-8 3-byte encoding
ASCII chars:          4,894,561  (85.8%)
Unique chars:               657
```

The core insight: **14% of characters consume 33% of bytes** under UTF-8.
A byte tokenizer therefore allocates one-third of its context window to
character-internal byte structure that carries no semantic signal.
BPE collapses this waste.

## Reproduce

```bash
# Step 1: train tokenizer
python train_tokenizer.py --data data/train_corpus.txt --merges 3840

# Step 2: train model (best config)
python train.py \
  --data data/train_corpus.txt \
  --steps 2000 \
  --out ckpt.pt \
  --optimizer muon --muon_lr 0.02 --lr 3e-3 \
  --n_embd 176 --n_layer 3 --n_head 8 \
  --block_size 384 --pos_enc rope \
  --batch 32 --schedule wsd --warmup 150 --decay_start 1600 \
  --ema --ema_decay 0.995 --ema_start 1500

# Step 3: evaluate
python evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
```
