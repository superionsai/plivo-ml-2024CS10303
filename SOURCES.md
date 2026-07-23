# Sources and References

This section documents the key techniques and their origins. All code is independently
implemented; no code was copied from external repositories.

## Techniques Investigated

### Byte-Pair Encoding (BPE) Tokenizer
- **Origin:** Sennrich et al., 2016 — "Neural Machine Translation of Rare Words with Subword Units"
  [ACL Anthology](https://aclanthology.org/P16-1162/)
- **Our adaptation:** Devanagari-aware pre-tokenization (U+0900-U+097F treated as atomic units)
  to ensure consonant+matra combinations merge correctly. Byte fallback for arbitrary UTF-8.

### Rotary Position Embedding (RoPE)
- **Origin:** Su et al., 2021 — "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **Why:** Frees ~(block_size × n_embd) parameters vs learned positional table.
  Encodes relative position through rotations, enabling better length generalization.

### RMSNorm
- **Origin:** Zhang & Sennrich, 2019 — "Root Mean Square Layer Normalization"
  [NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
- **Why:** Removes mean-centering bias term, fewer parameters, comparable quality to LayerNorm.

### Muon Optimizer (Momentum + Newton-Schulz Orthogonalization)
- **Origin:** Keller Jordan, modded-nanogpt (2024)
  [GitHub](https://github.com/KellerJordan/muon)
- **Core idea:** Orthogonalise the momentum buffer for 2D weight matrices via quintic
  Newton-Schulz iteration. This equalises singular value updates, making each of the
  fixed 2,000 steps more effective than AdamW alone.
- **Our implementation:** Hybrid optimizer — Muon for hidden 2D matrices, AdamW for
  embeddings, norms, and scalar/vector parameters. Independently implemented from the
  mathematical description.

### Warmup-Stable-Decay (WSD) Schedule
- **Origin:** Hu et al., 2025 — "MiniCPM: Unveiling the Potential of Small Language Models"
  [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6a1fe80a9e2dcda0b3e5fd0fd87eb097-Abstract-Conference.html)
- **Why:** Holds learning rate at peak longer than cosine, providing more useful gradient
  signal during the middle of training. Rapid improvement during final decay branch.

### EMA Weight Averaging
- **Origin:** Izmailov et al., 2018 — "Averaging Weights Leads to Wider Optima and Better Generalization"
  [BayesGroup](https://bayesgroup.org/publications/2018-averaging-weights-leads-to-wider-optima-and-better-generalization/)
- **Our use:** Late-start EMA (from step 1500) with decay 0.995, giving an effective
  averaging window of ~200 steps. Produces a smoother parameter estimate at zero
  additional compute.

### Weight Tying
- **Origin:** Press & Wolf, 2017 — "Using the Output Embedding to Improve Language Models"
  [EACL 2017](https://aclanthology.org/E17-2025/)
- **Implementation:** Functional tying — output logits computed via `F.linear(x, tok_emb.weight)`.
  Checkpoint stores one copy. Freed parameters reinvested in model width.

## Public Plivo Experiments Consulted
We reviewed public repositories from previous iterations of this assignment to understand
the competitive landscape and inform our hypotheses:
- Multiple BPE vocabulary sizes were tested across submissions (1024–8192)
- Muon was consistently the strongest optimizer choice
- Wider+shallower architectures outperformed deeper models within the step budget
- EMA provided consistent small improvements (~0.01-0.02 BPB)

## What We Tested But Did NOT Use
- **SwiGLU:** Parameter-matched SwiGLU gave no gain over GELU on this task scale —
  narrower MLP cancelled the activation benefit.
- **Deeper models (L=5+):** Underfit more; gradient signal dissipated across too many
  residual connections within 2,000 steps.
- **BPE-8192:** Diminishing compression returns; rare tokens get too few gradient updates.
- **Dropout > 0:** No benefit; corpus large enough relative to model size.
