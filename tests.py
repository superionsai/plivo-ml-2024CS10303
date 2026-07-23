"""Comprehensive correctness tests for tokenizer, model, and evaluation."""
import sys, os
sys.path.insert(0, '.')

def test_tokenizer():
    import tokenizer as tok_mod
    tok = tok_mod.load()
    print(f'Vocab size: {tok.vocab_size}')

    tests = {
        'empty':          '',
        'english':        'Hello, world! This is a test sentence.',
        'hindi':          '\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e! \u092f\u0939 \u090f\u0915 \u092a\u0930\u0940\u0915\u094d\u0937\u0923 \u0935\u093e\u0915\u094d\u092f \u0939\u0948\u0964',
        'code_mixed':     'Hello \u0928\u092e\u0938\u094d\u0924\u0947 world \u0926\u0941\u0928\u093f\u092f\u093e',
        'emoji':          '\U0001f600\U0001f389\U0001f525\U0001f4af\U0001f30d Hello \U0001f1ee\U0001f1f3',
        'cjk':            '\u4f60\u597d\u4e16\u754c \u3053\u3093\u306b\u3061\u306f \uc548\ub155\ud558\uc138\uc694',
        'combining':      '\u00e9 \u00e0 \u00fc \u00f1 \u00f6  cafe\u0301',
        'hindi_conjunct': '\u0915\u094d\u0937 \u0924\u094d\u0930 \u091c\u094d\u091e \u0936\u094d\u0930 \u0926\u094d\u0927',
        'tabs_newlines':  'line1\tcolA\tcolB\nline2\tcolC\tcolD\r\n',
        'whitespace':     '   multiple   spaces   ',
        'null_char':      'before\x00after',
        'all_ascii_printable': ''.join(chr(i) for i in range(32, 127)),
        'punctuation':    '!@#%^&*()_+-=[]{}|;:,.<>?/~`',
        'numbers':        '0123456789 3.14159 -42 1e10',
        'long_hindi':     '\u092d\u093e\u0930\u0924 \u090f\u0915 \u0935\u093f\u0935\u093f\u0927\u0924\u093e\u0913\u0902 \u0938\u0947 \u092d\u0930\u093e \u0926\u0947\u0936 \u0939\u0948',
        'mixed_script':   'The word \u092a\u094d\u0930\u0923\u093e\u092e means greeting',
        'repeated':       'aaaa' * 100,
    }

    passed = failed = 0
    for name, text in tests.items():
        try:
            encoded = tok.encode(text)
            decoded = tok.decode(encoded)
            if decoded == text:
                passed += 1
                bpt = len(text.encode('utf-8')) / max(len(encoded), 1) if encoded else 0
                print(f'  PASS {name:25s}  tokens={len(encoded):5d}  bytes/tok={bpt:.2f}')
            else:
                failed += 1
                for i, (a, b) in enumerate(zip(decoded, text)):
                    if a != b:
                        print(f'  FAIL {name:25s}  mismatch at char {i}: got {repr(a)} expected {repr(b)}')
                        break
                else:
                    print(f'  FAIL {name:25s}  length: {len(decoded)} vs {len(text)}')
        except Exception as e:
            failed += 1
            print(f'  FAIL {name:25s}  {e}')

    print(f'\n=== TOKENIZER: {passed}/{passed+failed} passed ===')
    return failed == 0


def test_model():
    import torch
    from model import GPT, Config

    cfg = Config()
    cfg.vocab_size = 4096
    cfg.block_size = 384
    cfg.n_embd = 176
    cfg.n_layer = 3
    cfg.n_head = 8
    cfg.tie_weights = True
    cfg.pos_enc = 'rope'
    model = GPT(cfg)

    print(f'\n--- Model Tests ---')
    n = model.n_params()
    print(f'  Params: {n:,}  (cap: 2,000,000)  {"PASS" if n <= 2_000_000 else "FAIL"}')

    # Weight tying: embedding and head must share the same tensor
    if cfg.tie_weights:
        tied = model.head is None  # functional tying means no separate head
        print(f'  Weight tying (functional): {"PASS" if tied else "FAIL"}')

    # Forward pass various lengths
    for T in [1, 2, 64, 128, 384]:
        x = torch.randint(0, cfg.vocab_size, (1, T))
        y = torch.randint(0, cfg.vocab_size, (1, T))
        logits, loss = model(x, y)
        ok = logits.shape == (1, T, cfg.vocab_size) and torch.isfinite(logits).all() and torch.isfinite(loss)
        print(f'  Forward T={T:3d}: shape={logits.shape}  finite={"PASS" if ok else "FAIL"}')

    # Causal test: changing future token must not affect past logits
    x1 = torch.randint(0, cfg.vocab_size, (1, 64))
    x2 = x1.clone()
    x2[0, 32] = (x2[0, 32] + 1) % cfg.vocab_size  # change token 32
    model.eval()
    with torch.no_grad():
        l1, _ = model(x1)
        l2, _ = model(x2)
    causal = torch.allclose(l1[0, :32], l2[0, :32], atol=1e-5)
    print(f'  Causal mask (future change does not affect past): {"PASS" if causal else "FAIL"}')

    # Checkpoint save/load roundtrip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmppath = f.name
    torch.save({
        'model': model.state_dict(),
        'config': {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_') and not callable(getattr(cfg, k))},
        'steps': 2000,
    }, tmppath)

    # Reload
    ckpt = torch.load(tmppath, map_location='cpu', weights_only=True)
    cfg2 = Config()
    for k, v in ckpt['config'].items():
        setattr(cfg2, k, v)
    model2 = GPT(cfg2)
    model2.load_state_dict(ckpt['model'])
    model2.eval()
    with torch.no_grad():
        l3, _ = model2(x1)
    reload_ok = torch.allclose(l1, l3, atol=1e-5)
    print(f'  Checkpoint reload logit equivalence: {"PASS" if reload_ok else "FAIL"}')
    n2 = model2.n_params()
    print(f'  Param count after reload: {n2:,} == {n:,}: {"PASS" if n2 == n else "FAIL"}')
    os.unlink(tmppath)

    return True


def test_compression_stats():
    import tokenizer as tok_mod
    tok = tok_mod.load()
    corpus = open('data/train_corpus.txt', encoding='utf-8').read()

    # English-only sample
    eng = ''.join(c for c in corpus[:300000] if ord(c) < 128)[:10000]
    hin = ''.join(c for c in corpus[:500000] if '\u0900' <= c <= '\u097F')[:10000]

    eng_enc = tok.encode(eng)
    hin_enc = tok.encode(hin)
    eng_bpt = len(eng.encode('utf-8')) / max(len(eng_enc), 1)
    hin_bpt = len(hin.encode('utf-8')) / max(len(hin_enc), 1)

    print(f'\n--- Compression by Script ---')
    print(f'  English: {eng_bpt:.3f} bytes/token')
    print(f'  Hindi:   {hin_bpt:.3f} bytes/token')

    # Full corpus stats
    full_enc = tok.encode(corpus[:100000])
    full_bpt = len(corpus[:100000].encode('utf-8')) / len(full_enc)
    print(f'  Mixed:   {full_bpt:.3f} bytes/token')

    # Token frequency analysis
    from collections import Counter
    token_counts = Counter(tok.encode(corpus[:500000]))
    total_unique = len(token_counts)
    rare_1_4 = sum(1 for c in token_counts.values() if c <= 4)
    rare_5_19 = sum(1 for c in token_counts.values() if 5 <= c <= 19)
    print(f'\n--- Token Frequency Analysis (first 500k chars) ---')
    print(f'  Unique tokens used: {total_unique}')
    print(f'  Rare (1-4 occurrences): {rare_1_4} ({100*rare_1_4/total_unique:.1f}%)')
    print(f'  Low  (5-19 occurrences): {rare_5_19} ({100*rare_5_19/total_unique:.1f}%)')


if __name__ == '__main__':
    tok_ok = test_tokenizer()
    test_model()
    test_compression_stats()
