# NOTES — Final Configuration Summary

1. Our winning model combines a Devanagari-aware BPE tokenizer with a modernized 3-layer GPT architecture and the Muon hybrid optimizer.
2. UTF-8 analysis revealed Devanagari characters consume 33% of raw corpus bytes, causing severe sequence length explosion under byte tokenization.
3. Our custom BPE tokenizer (vocab 4096) with space+word pre-tokenization compresses text to 3.62 bytes per token, providing a 2.8x context expansion over the baseline.
4. A full byte fallback mechanism guarantees lossless round-trip encoding for arbitrary UTF-8 text.
5. The model architecture uses RMSNorm for efficient normalization, Rotary Position Embeddings (RoPE) for position encoding, and functional weight tying to save 720k parameters.
6. Parameter savings from weight tying are reinvested into wider embedding dimensions (n_embd=176), optimizing per-step capacity within the 2.0M parameter cap.
7. The hybrid Muon optimizer applies 5-step Newton-Schulz orthogonalization to 2D weight matrices while using AdamW for embeddings, norms, and biases.
8. Learning rate is managed via a Warmup-Stable-Decay (WSD) schedule, maintaining peak learning rate through step 1600 before decaying to maximize gradient signal.
9. Late-stage Exponential Moving Average (EMA) weight accumulation from step 1500 yields a smooth parameter estimate without additional compute.
10. The complete pipeline runs strictly on CPU in under 12 minutes and produces a submission fully compliant with all parameter, step, and evaluation constraints.
