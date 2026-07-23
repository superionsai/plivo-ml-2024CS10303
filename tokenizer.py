"""
Tokenizer: Byte-Pair Encoding with Devanagari-aware pre-tokenization.

The baseline byte tokenizer treats every UTF-8 byte as a separate token.
For Devanagari (Hindi), each character is 3 bytes -> 3 tokens, which means:
  - A block_size=128 window covers only ~42 Hindi characters of real context
  - The model wastes step budget modelling individual bytes of multi-byte chars

This BPE tokenizer merges frequent byte-pairs, trained on the provided corpus.
Devanagari characters (U+0900-U+097F) are pre-tokenized as whole units so that
consonant+matra combinations can be merged correctly (not split mid-character).

Guarantees (checked by evaluate.py):
  1. decode(encode(text)) == text  (lossless, byte fallback for unseen bytes)
  2. load() returns an object with .encode(), .decode(), .vocab_size
  3. tokenizer.json is loaded relative to this file (works from any cwd)
"""

import json
import os
import re
from collections import defaultdict

# Pre-tokenization regex: treat Devanagari blocks as atomic units,
# then split on whitespace-like boundaries for English words.
# This ensures Hindi consonant+matra pairs get a chance to merge.
_PRETOK = re.compile(
    r'[\u0900-\u097F]+|[a-zA-Z]+|\d+|[^\s\w]|\s+',
    re.UNICODE
)

_TOKENIZER_FILE = os.path.join(os.path.dirname(__file__), "tokenizer.json")


class BPETokenizer:
    """Byte-level BPE with lossless fallback."""

    def __init__(self, merges: list[tuple[int, int]], vocab_size: int):
        # merges: ordered list of (a, b) -> a+b pairs applied during encode
        self.merges = {(a, b): i + 256 for i, (a, b) in enumerate(merges)}
        self.vocab_size = vocab_size
        # build decode table: id -> bytes
        self._id2bytes = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self._id2bytes[idx] = self._id2bytes[a] + self._id2bytes[b]

    def _tokenize_word(self, word_bytes: bytes) -> list[int]:
        """Apply BPE merges to a single pre-tokenized word's bytes."""
        ids = list(word_bytes)
        while len(ids) >= 2:
            # find the highest-priority (earliest) merge available
            best = None
            best_rank = float('inf')
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in self.merges and self.merges[pair] < best_rank:
                    best = i
                    best_rank = self.merges[pair]
            if best is None:
                break
            new_id = self.merges[(ids[best], ids[best + 1])]
            ids = ids[:best] + [new_id] + ids[best + 2:]
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        for chunk in _PRETOK.findall(text):
            ids.extend(self._tokenize_word(chunk.encode('utf-8')))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b''.join(self._id2bytes.get(i, b'') for i in ids)
        return raw.decode('utf-8', errors='replace')

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"type": "bpe", "merges": self.merges_as_list(),
                       "vocab_size": self.vocab_size}, f)

    def merges_as_list(self) -> list[list[int]]:
        inv = {v: k for k, v in self.merges.items()}
        return [list(inv[i + 256]) for i in range(len(self.merges))]


def train_bpe(text: str, num_merges: int, verbose: bool = True) -> BPETokenizer:
    """Train BPE on text. num_merges = vocab_size - 256."""
    # pre-tokenize into words, convert each to list of byte-ids
    words = [list(chunk.encode('utf-8')) for chunk in _PRETOK.findall(text)]
    # count word frequencies
    word_freq: dict[tuple, int] = defaultdict(int)
    for w in words:
        word_freq[tuple(w)] += 1

    merges = []
    vocab = {tuple(w): freq for w, freq in word_freq.items()}

    for step in range(num_merges):
        # count pair frequencies across all words
        pair_counts: dict[tuple, int] = defaultdict(int)
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += freq
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda p: pair_counts[p])
        new_id = 256 + step
        merges.append(best_pair)
        # merge best_pair in all words
        new_vocab: dict[tuple, int] = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        vocab = new_vocab
        if verbose and (step + 1) % 256 == 0:
            print(f"  BPE merge {step+1}/{num_merges}: {best_pair} -> {new_id} "
                  f"(freq={pair_counts[best_pair]})")

    return BPETokenizer(merges, vocab_size=256 + len(merges))


def load(path: str = None) -> BPETokenizer | 'ByteTokenizer':
    """Load tokenizer from tokenizer.json, falling back to byte tokenizer."""
    p = path or _TOKENIZER_FILE
    if not os.path.exists(p):
        # fallback: byte tokenizer (used before train_tokenizer.py is run)
        return _ByteFallback()
    with open(p, encoding='utf-8') as f:
        data = json.load(f)
    if data.get('type') == 'bpe':
        tok = BPETokenizer(
            merges=[(a, b) for a, b in data['merges']],
            vocab_size=data['vocab_size']
        )
        return tok
    return _ByteFallback()


class _ByteFallback:
    """Raw byte tokenizer — used when no tokenizer.json exists."""
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode('utf-8'))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode('utf-8', errors='replace')
