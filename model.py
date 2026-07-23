"""
GPT model — modernized from the starter baseline.

Key changes from baseline (each documented in RUNLOG.md):
  - RMSNorm instead of LayerNorm  : fewer params, empirically equivalent quality
  - RoPE positional encoding       : frees ~(block_size * n_embd) learned params,
                                     better length generalization
  - Functional weight tying        : output logits reuse tok_emb.weight (one copy
                                     in checkpoint), frees ~vocab_size*n_embd params
                                     for wider/deeper architecture under the 2M cap
  - GPT-2-style residual scaling   : std = 0.02 / sqrt(2 * n_layer) on output
                                     projections prevents variance explosion with depth
  - Causal mask via F.sdpa         : already in baseline, kept as-is

Architecture search (see RUNLOG): wider+shallower wins on this corpus size.
Best found: n_embd=176, n_layer=3, n_head=8, block_size=384, vocab~4096
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size: int = 256       # overridden after tokenizer is loaded
    block_size: int = 384       # context length (tokens)
    n_layer: int = 3
    n_head: int = 8
    n_embd: int = 176
    dropout: float = 0.0
    tie_weights: bool = True    # functional weight tying
    pos_enc: str = "rope"       # "rope" | "learned"


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean-subtraction bias)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32 for numerical stability, then back
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ── Rotary Position Embedding ────────────────────────────────────────────────

def _build_rope_cache(seq_len: int, head_dim: int,
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float()
                              / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)           # (T, head_dim/2)
    cos = freqs.cos()[None, None, :, :]             # (1,1,T,head_dim/2)
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to query or key tensor (B, H, T, head_dim)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ── Attention ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = cfg.pos_enc == "rope"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        # rope cache (rebuilt if seq_len changes)
        self._rope_len = 0
        self._cos = self._sin = None

    def _get_rope(self, T: int, device: torch.device):
        if T > self._rope_len:
            self._cos, self._sin = _build_rope_cache(T, self.head_dim, device)
            self._rope_len = T
        return self._cos[:, :, :T, :], self._sin[:, :, :T, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # reshape to (B, H, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            cos, sin = self._get_rope(T, x.device)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.out_proj(y))


# ── MLP ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Standard 4x MLP. GELU activation (baseline kept)."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp   = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── GPT ──────────────────────────────────────────────────────────────────────

class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # learned positional embedding only when not using RoPE
        if cfg.pos_enc == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        else:
            self.pos_emb = None
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f  = RMSNorm(cfg.n_embd)
        # head is a plain Linear; when tie_weights=True we use the embedding
        # matrix functionally (no double-counting in n_params)
        if not cfg.tie_weights:
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        else:
            self.head = None
        self._init_weights()

    def _init_weights(self):
        """GPT-2-style init: N(0, 0.02), residual projections scaled by depth."""
        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = 0.02
                if "out_proj" in name or "fc2" in name:
                    std /= math.sqrt(2 * self.cfg.n_layer)
                nn.init.normal_(p, mean=0.0, std=std)
            elif p.ndim == 1:
                nn.init.zeros_(p)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor = None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        # functional weight tying: reuse embedding matrix as output projection
        if self.cfg.tie_weights:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss

    def n_params(self) -> int:
        """Parameter count — tied weights counted once."""
        return sum(p.numel() for p in self.parameters())
