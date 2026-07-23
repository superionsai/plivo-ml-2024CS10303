# Run 0 — Baseline (starter code, unchanged)
**Corpus stats (our own measurement):**
- Total bytes: 7,318,592
- Total chars: 5,703,936
- Devanagari chars: 801,846 (14.1% of chars → 32.9% of bytes)
- Bytes per Devanagari char: 3.00
- ASCII chars: 4,894,561 (85.8%)

**Observation:** The byte tokenizer wastes ~33% of the corpus budget on Devanagari byte triplets.
A block_size=128 window covers only ~42 Hindi characters of real context. This is the primary bottleneck.

**Config:** byte vocab=256, block=128, L=4/H=4/E=160, Adam lr=3e-4, batch=8, no schedule, no warmup, no weight tying, no grad clip
**Result:** Dev BPB = **2.3718** (baseline)
**Time:** ~73s

**Conclusion:** Confirmed. The model is step-limited — loss still clearly descending at step 2000.
Three structural problems: (1) byte tokenizer on Hindi, (2) no LR schedule, (3) tiny batch.

---

# Run 1 — Fix obvious training hygiene (no architecture change)
**Hypothesis:** Batch 8→32, add cosine LR schedule + warmup + grad clipping. These are table stakes.
**Config:** same arch, AdamW lr=3e-3 (10x increase + schedule), batch=32, warmup=150, cosine, clip=1.0, weight tying=True
**Expected:** ~1.9-2.1 BPB

---

# Run 2 — BPE tokenizer (vocab 2048, our own implementation)
**Hypothesis:** Devanagari-aware BPE should compress 3 bytes/char -> ~1 token/char, giving 3x more
real context per step. vocab=2048 = 256 bytes + 1792 merges.
**Key design choice:** pre-tokenize Devanagari block (U+0900-U+097F) as atomic units before BPE,
so consonant+matra pairs can merge correctly.
**Expected:** ~1.70-1.80 BPB (tokenizer alone is the biggest single win)

---

# Run 3 — BPE vocab sweep (1024 vs 2048 vs 4096)
**Hypothesis:** More merges = better compression but rarer tokens get fewer updates in 2000 steps.
Find the sweet spot for this corpus + step budget.
**Configs (all else equal):** vocab 1024, 2048, 4096
**Decision criteria:** lower dev BPB wins; tie goes to smaller vocab (easier to train)

---

# Run 4 — Architecture: RMSNorm + RoPE + weight tying
**Hypothesis:** Replace LayerNorm -> RMSNorm (fewer params, same quality), absolute pos -> RoPE
(frees vocab_size*n_embd params, better length gen), ensure functional weight tying.
**Expected delta:** ~0.02-0.03 BPB improvement over Run 2

---

# Run 5 — Width vs Depth: E176/L3 vs E192/L3 vs E176/L4
**Hypothesis:** Wider+shallower wins on small corpus (better gradient flow, less residual depth needed).
Under 2M param cap with BPE-4096 + weight tying, available budget changes significantly.
**Configs:** vary (n_embd, n_layer) at equal param count

---

# Run 6 — Muon optimizer
**Hypothesis:** Muon orthogonalises momentum for 2D weight matrices, equalising singular value updates.
With only 2000 steps, better per-step progress = directly better final BPB.
Use hybrid: Muon for matmuls, AdamW for embeddings/norms/biases.
**muon_lr=0.02, adamw_lr=3e-3**
**Expected delta:** ~0.02-0.03 BPB improvement

---

# Run 7 — Context window: block_size 128 -> 256 -> 384
**Hypothesis:** Under Muon's faster convergence, longer context can actually be used within 2000 steps.
Under AdamW this was inert; under Muon it may pay off.
**Expected:** 384 > 256 > 128 (under Muon only)

---

# Run 8 — WSD schedule vs cosine
**Hypothesis:** Warmup-Stable-Decay holds LR at peak longer before decaying, giving more useful
gradient signal during the middle of training.
**Config:** warmup=150, stable until step 1600, cosine decay 1600->2000

---

# Run 9 — EMA weight averaging
**Hypothesis:** Averaging model weights over last ~500 steps (starting at step 1500) gives a
smoother parameter estimate that generalises better than the final noisy checkpoint.
**ema_decay=0.995, ema_start=1500**
**Expected:** ~0.005-0.015 BPB improvement at no computational cost

---

# Run 10 — Best config, final run
**Config:** (filled after ablations)
**Dev BPB:** (TBD)
