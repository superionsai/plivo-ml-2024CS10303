from __future__ import annotations
"""
Tokenizer: fast Byte-Pair Encoding with Devanagari-aware pre-tokenization.

The baseline byte tokenizer treats every UTF-8 byte as a separate token.
For Devanagari (Hindi), each character is 3 bytes -> 3 tokens, which means:
  - A block_size=128 window covers only ~42 Hindi characters of real context
  - The model wastes step budget modelling individual bytes of multi-byte chars

This BPE tokenizer merges frequent byte-pairs, trained on the provided corpus.
Devanagari characters (U+0900-U+097F) are pre-tokenized as whole units so that
consonant+matra combinations can be merged correctly (not split mid-character).

Training uses incremental pair-count updates (only words containing the merged
pair are rescanned per step) — significantly faster than full-corpus rescans.

Guarantees (checked by evaluate.py):
  1. decode(encode(text)) == text  (lossless, byte fallback for unseen bytes)
  2. load() returns object with .encode(), .decode(), .vocab_size
  3. tokenizer.json is loaded relative to this file (works from any cwd)
"""

import json
import os
import re
from collections import defaultdict

# Allow space+word merges: ` ?[a-zA-Z]+` lets the leading space join the word,
# enabling BPE to learn tokens like ' the', ' and', ' is' — the single biggest
# compression improvement. Devanagari stays atomic for correct conjunct merging.
_PRETOK = re.compile(r'[\u0900-\u097F]+| ?[a-zA-Z]+| ?\d+|\s+|.', re.UNICODE | re.DOTALL)
_TOKENIZER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")


class BPETokenizer:
    """Byte-level BPE with lossless byte fallback."""

    def __init__(self, merges: list[tuple[int, int]], vocab_size: int):
        self.merges = {(a, b): i + 256 for i, (a, b) in enumerate(merges)}
        self.vocab_size = vocab_size
        self._id2bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self._id2bytes[idx] = self._id2bytes[a] + self._id2bytes[b]
        self._encode_cache: dict[str, list[int]] = {}  # word→token-ids cache

    def _apply_merges(self, ids: list[int]) -> list[int]:
        """Apply all BPE merges greedily. O(n^2) per call but cached at word level."""
        while len(ids) >= 2:
            best_rank, best_i = float('inf'), -1
            for i in range(len(ids) - 1):
                r = self.merges.get((ids[i], ids[i+1]), float('inf'))
                if r < best_rank:
                    best_rank, best_i = r, i
            if best_i == -1:
                break
            ids = ids[:best_i] + [self.merges[(ids[best_i], ids[best_i+1])]] + ids[best_i+2:]
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        cache = self._encode_cache
        for chunk in _PRETOK.findall(text):
            if chunk in cache:
                ids.extend(cache[chunk])
            else:
                encoded = self._apply_merges(list(chunk.encode('utf-8')))
                cache[chunk] = encoded
                ids.extend(encoded)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b''.join(self._id2bytes.get(i, b'') for i in ids).decode('utf-8', errors='replace')

    def save(self, path: str):
        inv = {v: k for k, v in self.merges.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"type": "bpe",
                       "merges": [list(inv[256 + i]) for i in range(len(self.merges))],
                       "vocab_size": self.vocab_size}, f)


def train_bpe(text: str, num_merges: int, verbose: bool = True) -> BPETokenizer:
    """
    Train BPE. Uses per-word scanning only for words that contain the merged
    pair at each step — much faster than full-corpus rescan.

    Strategy:
      - word_ids[i]: current token list for word i
      - pair_to_words[pair]: set of word indices containing that pair
      - pair_count[pair]: total frequency of that pair across all words
    """
    import time
    t0 = time.time()

    # Pre-tokenize: build (word_bytes, freq) table
    word_freq: dict[bytes, int] = defaultdict(int)
    for chunk in _PRETOK.findall(text):
        word_freq[chunk.encode('utf-8')] += 1

    # Convert to indexed structure
    words_bytes = list(word_freq.keys())
    words_freq  = [word_freq[w] for w in words_bytes]
    word_ids    = [list(w) for w in words_bytes]  # mutable token lists
    n_words     = len(word_ids)

    # Build pair_count and pair_to_words index
    pair_count: dict[tuple, int] = defaultdict(int)
    pair_to_words: dict[tuple, set] = defaultdict(set)

    for wi, (ids, freq) in enumerate(zip(word_ids, words_freq)):
        for j in range(len(ids) - 1):
            p = (ids[j], ids[j+1])
            pair_count[p] += freq
            pair_to_words[p].add(wi)

    merges: list[tuple[int, int]] = []

    for step in range(num_merges):
        if not pair_count:
            break
        # Find best pair (max frequency)
        best_pair = max(pair_count, key=lambda p: pair_count[p])
        if pair_count[best_pair] <= 0:
            break

        new_id = 256 + step
        merges.append(best_pair)
        a, b = best_pair

        # Only rescan words that contain this pair
        affected = list(pair_to_words.get(best_pair, set()))
        for wi in affected:
            ids  = word_ids[wi]
            freq = words_freq[wi]
            new_ids: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i+1] == b:
                    # Remove adjacency pairs being destroyed
                    if new_ids:
                        old_l = (new_ids[-1], a)
                        pair_count[old_l] -= freq
                        pair_to_words[old_l].discard(wi)
                    if i + 2 < len(ids):
                        old_r = (b, ids[i+2])
                        pair_count[old_r] -= freq
                        pair_to_words[old_r].discard(wi)
                    # Add new adjacency pairs being created
                    if new_ids:
                        new_l = (new_ids[-1], new_id)
                        pair_count[new_l] += freq
                        pair_to_words[new_l].add(wi)
                    if i + 2 < len(ids):
                        new_r = (new_id, ids[i+2])
                        pair_count[new_r] += freq
                        pair_to_words[new_r].add(wi)
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            word_ids[wi] = new_ids

        # Remove the merged pair from tracking
        del pair_count[best_pair]
        del pair_to_words[best_pair]

        if verbose and (step + 1) % 512 == 0:
            el = time.time() - t0
            print(f"  BPE merge {step+1}/{num_merges}  "
                  f"({el:.1f}s, {el/(step+1)*1000:.0f}ms/merge)")

    return BPETokenizer(merges, vocab_size=256 + len(merges))


def load(path: str = None):
    """Load BPE tokenizer from tokenizer.json, falling back to byte tokenizer."""
    p = path or _TOKENIZER_FILE
    if not os.path.exists(p):
        return _ByteFallback()
    with open(p, encoding='utf-8') as f:
        data = json.load(f)
    if data.get('type') == 'bpe':
        return BPETokenizer(merges=[(a, b) for a, b in data['merges']],
                            vocab_size=data['vocab_size'])
    return _ByteFallback()


class _ByteFallback:
    """Raw byte tokenizer — fallback when no tokenizer.json is present."""
    vocab_size = 256
    def encode(self, text: str) -> list[int]: return list(text.encode('utf-8'))
    def decode(self, ids: list[int]) -> str: return bytes(ids).decode('utf-8', errors='replace')
