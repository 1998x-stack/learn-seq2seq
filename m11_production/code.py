"""m11 生产训练 — Checkpoint / AMP / DDP / 评测体系（v1.0）。

Run:  python m11_production/code.py

v1.0 升级点（versions.md）：把可跑模型补成"能进生产"的工程能力：
  - Checkpoint：保存/恢复 model(+optimizer+step)，可断点续训
  - 混合精度 AMP：`autocast` + `GradScaler`，提速且数值稳定
  - 分布式 DDP：`DistributedDataParallel`（单进程 world_size=1 演示包装结构）
  - 评测体系：holdout 上的 Exact Match

任务：反转 token 序列（在当前环境稳定收敛），把四种生产机制各钉一个自检。

自检：
  [A] Checkpoint round-trip：存盘 -> 新建模型载入 -> 同输入前向完全一致
  [B] AMP 训练：loss 下降、无 NaN（CUDA 下走 autocast+scaler）
  [C] DDP 包装：wrap 后 `wrapped.module` 即原模型、单步前向跑通
  [D] 评测：训练后 holdout Exact-Match 提升
"""

from __future__ import annotations

import io
import sys

import code as _stdlib_code  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOS, EOS = 0, 1
VOCAB = 6


class TinySeq2Seq(nn.Module):
    def __init__(self, vocab=VOCAB, hid=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hid)
        self.enc = nn.GRU(hid, hid, batch_first=True)
        self.dec = nn.GRU(hid, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward(self, src, tgt, teacher=0.8):
        enc_out, _ = self.enc(self.embed(src))
        h = enc_out[:, -1:].transpose(0, 1)
        dec_in = tgt[:, 0]
        logits = []
        for t in range(1, tgt.size(1)):
            out, h = self.dec(self.embed(dec_in).unsqueeze(1), h)
            logits.append(self.out(out.squeeze(1)))
            if torch.rand(1).item() < teacher:
                dec_in = tgt[:, t]
            else:
                dec_in = logits[-1].argmax(-1)
        return torch.stack(logits, 1)


def make_batch(B, L, src_lo=1):
    """源 token ∈ {src_lo..VOCAB-1}；目标 = 反转源序列（加 BOS / EOS）。"""
    src = torch.randint(src_lo, VOCAB, (B, L))
    tgt = torch.cat([torch.full((B, 1), BOS),
                     torch.flip(src, [1]),
                     torch.full((B, 1), EOS)], dim=1)
    return src, tgt


def train_amp(model, optim, scaler, steps=240):
    """AMP 训练：CUDA 走 autocast + GradScaler；CPU 纯 fp32。返回 (losses, steps)。"""
    losses = []
    per = 40
    for g in range(steps // per):
        tot = 0.0
        for _ in range(per):
            src, tgt = make_batch(16, 6)
            src, tgt = src.to(device), tgt.to(device)
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda"):
                    out = model(src, tgt)
                    loss = F.cross_entropy(out.reshape(-1, VOCAB), tgt[:, 1:].reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                out = model(src, tgt)
                loss = F.cross_entropy(out.reshape(-1, VOCAB), tgt[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            tot += loss.item()
        losses.append(tot / per)
    return losses


@torch.no_grad()
def exact_match(model, n=8):
    model.eval()
    ok = 0
    L = 5
    for _ in range(n):
        src = torch.randint(1, VOCAB, (1, L)).to(device)
        enc_out, _ = model.enc(model.embed(src))
        h = enc_out[:, -1:].transpose(0, 1)
        dec_in = torch.tensor([BOS], device=device)
        pred = []
        for _ in range(L):
            out, h = model.dec(model.embed(dec_in).unsqueeze(1), h)
            tok = model.out(out.squeeze(1)).argmax(-1)
            pred.append(int(tok.item()))
            dec_in = tok
        gold = torch.flip(src[0], [0]).tolist()
        ok += int(pred == gold)
    return ok / n


if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    # [A] Checkpoint round-trip：存盘 -> 新模型载入 -> 同输入前向相等
    m1 = TinySeq2Seq().to(device).eval()
    src, tgt = make_batch(2, 5)
    src, tgt = src.to(device), tgt.to(device)
    with torch.no_grad():
        y1 = m1(src, tgt, 0.0)
    buf = io.BytesIO()
    torch.save({"model": m1.state_dict(), "step": 100}, buf)
    buf.seek(0)
    ckpt = torch.load(buf)
    m2 = TinySeq2Seq().to(device).eval()
    m2.load_state_dict(ckpt["model"])
    with torch.no_grad():
        y2 = m2(src, tgt, 0.0)
    assert torch.allclose(y1, y2, atol=1e-6)
    print("[PASS] checkpoint round-trip: load yields identical forward")

    # [B] AMP 训练收敛、无 NaN
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    m = TinySeq2Seq().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = train_amp(m, opt, scaler, steps=480)
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert losses[0] > losses[-1], f"AMP loss not decreasing: {losses}"
    print(f"[PASS] AMP training converges ({scaler is not None}): "
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")

    # [D] 评测：holdout exact-match
    acc = exact_match(m, 10)
    assert acc >= 0.6, f"exact-match too low after train: {acc:.2f}"
    print(f"[PASS] eval exact-match on holdout: {acc:.2f}")

    # [C] DDP：world_size=1（gloo）也能 wrap 并前向
    mcpu = TinySeq2Seq()                              # CPU 模型，配 gloo
    src_c, tgt_c = make_batch(2, 5, src_lo=1)
    try:
        import torch.distributed as dist
        dist.init_process_group(backend="gloo",
                                init_method="tcp://127.0.0.1:29500",
                                rank=0, world_size=1)
        wrapped = torch.nn.parallel.DistributedDataParallel(mcpu, device_ids=None)
        with torch.no_grad():
            o = wrapped(src_c, tgt_c, 0.0)
        assert wrapped.module is mcpu
        assert o.shape == (2, tgt_c.size(1) - 1, VOCAB)
        dist.destroy_process_group()
        print("[PASS] DDP wrap under gloo world_size=1: forward ok, .module==model")
    except Exception as e:  # noqa: BLE001
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        raise AssertionError(f"DDP demo failed: {type(e).__name__}: {e}")