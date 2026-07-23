"""
Train the BPE tokenizer on the provided corpus and save tokenizer.json.

Usage:
    python train_tokenizer.py --data data/train_corpus.txt --merges 3840

vocab_size = 256 (byte base) + merges
We sweep: 768, 1792, 3840 merges -> vocab 1024, 2048, 4096
"""
import argparse
import time
import tokenizer as tok_mod


def measure_compression(tok, text_sample: str) -> float:
    """Bytes per token on a sample — higher means better compression."""
    n_bytes = len(text_sample.encode('utf-8'))
    n_tokens = len(tok.encode(text_sample))
    return n_bytes / n_tokens


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_corpus.txt")
    ap.add_argument("--merges", type=int, default=3840,
                    help="num BPE merges (vocab_size = 256 + merges)")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    text = open(args.data, encoding='utf-8').read()
    sample = text[:50_000]  # quick compression check on subset

    print(f"Training BPE: {args.merges} merges -> vocab {256 + args.merges}")
    t0 = time.time()
    tok = tok_mod.train_bpe(text, num_merges=args.merges, verbose=not args.quiet)
    print(f"Done in {time.time()-t0:.1f}s")
    print(f"Compression on sample: {measure_compression(tok, sample):.3f} bytes/token")

    # verify losslessness on a diverse sample covering all unique chars
    # (use first 200k chars which includes all unique character types in corpus)
    check_sample = text[:200_000]
    encoded = tok.encode(check_sample)
    decoded = tok.decode(encoded)
    assert decoded == check_sample, (
        f"tokenizer is not lossless! "
        f"First mismatch at char {next(i for i,(a,b) in enumerate(zip(decoded,check_sample)) if a!=b)}"
    )
    print("Lossless check: PASS")

    tok.save(args.out)
    print(f"Saved -> {args.out}  (vocab_size={tok.vocab_size})")
