"""
Training script for the 2,000-step LLM speedrun.

Hard caps (enforced here and at grading):
  * steps  <= 2,000
  * params <= 2,000,000
  * data   = train_corpus.txt only
  * pure PyTorch / numpy / stdlib

Usage examples:
  # baseline-style run (sanity check):
  python train.py --data data/train_corpus.txt --steps 2000 --out ckpt.pt

  # full run with best config:
  python train.py --data data/train_corpus.txt --steps 2000 --out ckpt.pt \\
      --optimizer muon --muon_lr 0.02 --lr 3e-3 \\
      --n_embd 176 --n_layer 3 --n_head 8 --block_size 384 \\
      --batch 32 --schedule wsd --warmup 150 --decay_start 1600 \\
      --ema --ema_decay 0.995 --ema_start 1500

Flags:
  --optimizer  : "adamw" (baseline) or "muon" (fast matrix convergence)
  --schedule   : "cosine" or "wsd" (warmup-stable-decay)
  --ema        : enable EMA weight averaging for evaluation
  --ema_start  : step at which EMA accumulation begins
"""
import argparse
import copy
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod
from optim import MuonOptimizer

MAX_STEPS  = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids: torch.Tensor, block: int, batch: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:     i + block    ] for i in ix])
    y = torch.stack([ids[i + 1: i + block + 1] for i in ix])
    return x.to(device), y.to(device)


def count_tokens(text: str, tok) -> int:
    return len(tok.encode(text[:100_000]))  # sample for speed


def main():
    ap = argparse.ArgumentParser()
    # data / output
    ap.add_argument("--data",       required=True)
    ap.add_argument("--out",        default="ckpt.pt")
    ap.add_argument("--steps",      type=int,   default=2000)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--log_every",  type=int,   default=100)

    # architecture
    ap.add_argument("--n_embd",     type=int,   default=176)
    ap.add_argument("--n_layer",    type=int,   default=3)
    ap.add_argument("--n_head",     type=int,   default=8)
    ap.add_argument("--block_size", type=int,   default=384)
    ap.add_argument("--pos_enc",    default="rope", choices=["rope", "learned"])
    ap.add_argument("--no_tie",     action="store_true", default=False,
                    help="disable weight tying (default: tying ON)")

    # optimiser
    ap.add_argument("--optimizer",  default="muon", choices=["adamw", "muon"])
    ap.add_argument("--lr",         type=float, default=3e-3,
                    help="AdamW LR (or sole LR if --optimizer adamw)")
    ap.add_argument("--muon_lr",    type=float, default=0.02,
                    help="Muon LR for 2D matrix weights")
    ap.add_argument("--wd",         type=float, default=0.1)
    ap.add_argument("--batch",      type=int,   default=32)
    ap.add_argument("--clip",       type=float, default=1.0)

    # schedule
    ap.add_argument("--schedule",   default="wsd", choices=["cosine", "wsd"])
    ap.add_argument("--warmup",     type=int,   default=150)
    ap.add_argument("--decay_start",type=int,   default=1600,
                    help="WSD: step where decay begins (ignored for cosine)")

    # EMA
    ap.add_argument("--ema",        action="store_true", default=False,
                    help="EMA weight averaging for the saved checkpoint (recommended)")
    ap.add_argument("--ema_decay",  type=float, default=0.995)
    ap.add_argument("--ema_start",  type=int,   default=1500,
                    help="step from which EMA accumulation starts")

    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok = tokenizer_mod.load()
    text = open(args.data, encoding="utf-8").read()
    ids  = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus : {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size}, "
          f"{len(text.encode('utf-8'))/len(ids):.2f} bytes/tok)")

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = Config()
    cfg.vocab_size  = tok.vocab_size
    cfg.block_size  = args.block_size
    cfg.n_layer     = args.n_layer
    cfg.n_head      = args.n_head
    cfg.n_embd      = args.n_embd
    cfg.pos_enc     = args.pos_enc
    cfg.tie_weights = not args.no_tie

    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model  : {n:,} params  "
          f"(arch: L={cfg.n_layer} H={cfg.n_head} E={cfg.n_embd} "
          f"ctx={cfg.block_size} tied={cfg.tie_weights})")
    assert n <= MAX_PARAMS, f"cap exceeded: {n:,} > {MAX_PARAMS:,}"

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if args.optimizer == "muon":
        opt = MuonOptimizer(
            model.parameters(),
            muon_lr   = args.muon_lr,
            adamw_lr  = args.lr,
            momentum  = 0.95,
            adamw_wd  = args.wd,
        )
        def _lr_at(step: int) -> tuple[float, float]:
            if args.schedule == "wsd":
                lr_m = MuonOptimizer.wsd_lr(step, args.steps, args.warmup,
                                             args.decay_start, args.muon_lr)
                lr_a = MuonOptimizer.wsd_lr(step, args.steps, args.warmup,
                                             args.decay_start, args.lr)
            else:
                lr_m = MuonOptimizer.cosine_lr(step, args.steps,
                                                args.warmup, args.muon_lr)
                lr_a = MuonOptimizer.cosine_lr(step, args.steps,
                                                args.warmup, args.lr)
            return lr_m, lr_a
    else:
        # plain AdamW baseline path
        _adamw = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.wd, betas=(0.9, 0.95))
        _sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            _adamw, T_max=args.steps - args.warmup, eta_min=args.lr * 0.1)

    # ── EMA shadow model ──────────────────────────────────────────────────────
    ema_model = None
    if args.ema:
        ema_model  = copy.deepcopy(model)
        ema_n_acc  = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    losses: list[float] = []

    for step in range(1, args.steps + 1):
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)

        if args.optimizer == "muon":
            opt.zero_grad(set_to_none=True)
        else:
            _adamw.zero_grad(set_to_none=True)

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if args.optimizer == "muon":
            lr_m, lr_a = _lr_at(step)
            opt.step(muon_lr=lr_m, adamw_lr=lr_a)
        else:
            # warmup phase for AdamW
            if step <= args.warmup:
                for pg in _adamw.param_groups:
                    pg['lr'] = args.lr * step / args.warmup
            else:
                _sched.step()
            _adamw.step()

        losses.append(loss.item())

        # EMA accumulation
        if args.ema and ema_model is not None and step >= args.ema_start:
            with torch.no_grad():
                for p_ema, p_model in zip(ema_model.parameters(),
                                          model.parameters()):
                    p_ema.data.mul_(args.ema_decay).add_(
                        p_model.data, alpha=1.0 - args.ema_decay)
            ema_n_acc += 1

        if step % args.log_every == 0 or step == 1:
            window = losses[-args.log_every:]
            avg = sum(window) / len(window)
            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            if args.optimizer == "muon":
                lr_m, lr_a = _lr_at(step)
                lr_str = f"lr_muon={lr_m:.5f} lr_adamw={lr_a:.5f}"
            else:
                lr_str = f"lr={_adamw.param_groups[0]['lr']:.5f}"
            print(f"step {step:5d}  loss {avg:.4f}  {lr_str}  "
                  f"({ms_per:.0f} ms/step  {elapsed:.0f}s total)")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    # Use EMA weights in saved model if available, else use final model weights
    save_model = ema_model if (args.ema and ema_model is not None
                               and ema_n_acc > 0) else model
    save_model.eval()
    torch.save({
        "model"            : save_model.state_dict(),
        "config"           : {k: getattr(cfg, k)
                              for k in dir(cfg)
                              if not k.startswith("_")
                              and not callable(getattr(cfg, k))},
        "steps"            : args.steps,
        "train_loss_curve" : losses,
        "ema_steps"        : ema_n_acc if args.ema else 0,
    }, args.out)
    print(f"\nsaved {args.out}  ({time.time()-t0:.0f}s total)")
    print(f"EMA: averaged over last {ema_n_acc} steps" if ema_n_acc else "EMA: not used")


if __name__ == "__main__":
    main()
