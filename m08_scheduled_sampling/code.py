"""m08 Scheduled Sampling + Label Smoothing — 对齐训练/推理分布、降过度自信（v0.7）。

Run:  python m08_scheduled_sampling/code.py

v0.7 升级点（versions.md）：
  - Scheduled Sampling：训练时以越来越低的概率用真值（Teacher Forced），
    越来越高地用模型自预测，缓解 Exposure Bias（train 喂真、infer 喂自己）。
  - Label Smoothing：把 one-hot 目标抹平到 epsilon，防止模型 drive 过自信。

自检（机制级，不依赖网络收敛好坏）：
  [A] scheduled_ratio(epoch) 单调下降、有下界            —— 调度正确
  [B] label_smoothing 的 CE 恒等式：手动展开 == nn.CrossEntropyLoss(=smoothing)
  [C] 平滑后的目标分布 sum==1 且金标质量从 epsilon 抹到 1-epsilon
  [D] 配平滑的深层 seq2seq 训练 loss 有限且下降           —— 工程可用
"""

from __future__ import annotations

import math
import random
import sys
from typing import List

import code as _stdlib_code  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ZH = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
mapping = dict(zip(EN, ZH))


class Vocabulary:
    def __init__(self, tokens: List[str]):
        self.t2i = {t: i for i, t in enumerate(SPECIAL_TOKENS + tokens)}
        self.i2t = {i: t for t, i in self.t2i.items()}
        self.pad = self.t2i["<PAD>"]; self.bos = self.t2i["<BOS>"]
        self.eos = self.t2i["<EOS>"]; self.unk = self.t2i["<UNK>"]

    def encode(self, toks, add_bos=False, add_eos=False):
        ids = [self.t2i.get(t, self.unk) for t in toks]
        if add_bos: ids = [self.bos] + ids
        if add_eos: ids = ids + [self.eos]
        return ids

    def __len__(self): return len(self.t2i)


src_vocab = Vocabulary(EN)
tgt_vocab = Vocabulary(ZH)


# ----------------------------------------------------------------------------
# 1. Scheduled Sampling 调度（v0.7）
# ----------------------------------------------------------------------------

def scheduled_ratio(epoch: int, initial: float = 1.0,
                    minimum: float = 0.0, decay: float = 0.9) -> float:
    """tf 概率随 epoch 指数衰减，且不低于 minimum。"""
    r = initial * (decay ** (epoch - 1))
    return max(minimum, r)


# ----------------------------------------------------------------------------
# 2. 深度 server 网络（含 label_smoothing 的 CE）
# ----------------------------------------------------------------------------

def make_example():
    length = random.randint(1, 6)
    src = [random.choice(EN) for _ in range(length)]
    return src, [mapping[t] for t in src]


class NumberDataset(Dataset):
    def __init__(self, size):
        self.examples = [make_example() for _ in range(size)]

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        src, tgt = self.examples[idx]
        s = torch.tensor(src_vocab.encode(src, add_eos=True))
        t = torch.tensor(tgt_vocab.encode(tgt, add_bos=True, add_eos=True))
        return s, t, len(s)


def collate(examples):
    srcs = [e[0] for e in examples]; lens = [e[2] for e in examples]; S = max(lens)
    B = len(examples)
    src = torch.zeros(B, S, dtype=torch.long).fill_(src_vocab.pad)
    for r, (s, l) in enumerate(zip(srcs, lens)):
        src[r, :l] = s
    tgt = pad_sequence([e[1] for e in examples], batch_first=True,
                       padding_value=tgt_vocab.pad)
    return src, torch.tensor(lens), tgt


def build_loader(size, bs):
    return DataLoader(NumberDataset(size), batch_size=bs, shuffle=True,
                      collate_fn=collate)


class Seq2Seq(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.hembed = nn.Embedding(len(src_vocab), hid, padding_idx=src_vocab.pad)
        self.enc = nn.GRU(hid, hid, batch_first=True)
        self.dec_emb = nn.Embedding(len(tgt_vocab), hid, padding_idx=tgt_vocab.pad)
        self.dec = nn.GRU(hid, hid, batch_first=True)
        self.linear = nn.Linear(hid, len(tgt_vocab))

    def forward(self, src, tgt, tf_ratio):
        enc_out = self.enc(self.hembed(src))[0]
        h = enc_out[:, -1:].transpose(0, 1)          # [1,B,H]
        dec_in = tgt[:, 0]
        logits = []
        for t in range(1, tgt.size(1)):
            out, h = self.dec(self.dec_emb(dec_in).unsqueeze(1), h)
            log = self.linear(out.squeeze(1))
            logits.append(log)
            if random.random() < tf_ratio:
                dec_in = tgt[:, t]
            else:
                dec_in = log.argmax(-1)
        return torch.stack(logits, 1)


def run_epoch(model, loader, optim, tf, smoothing):
    model.train()
    total, n = 0.0, 0
    for src, lens, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optim.zero_grad(set_to_none=True)
        out = model(src, tgt, tf)
        loss = F.cross_entropy(out.reshape(-1, len(tgt_vocab)),
                               tgt[:, 1:].reshape(-1),
                               ignore_index=tgt_vocab.pad,
                               label_smoothing=smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    # [A] Scheduled Sampling 调度单调衰减且有界
    ratios = [scheduled_ratio(e, initial=1.0, decay=0.8, minimum=0.2)
              for e in range(1, 9)]
    assert all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1))
    assert 0.0 <= ratios[-1] <= 1.0
    assert ratios[0] == 1.0
    print(f"[PASS] scheduled_ratio decays: {[round(r, 2) for r in ratios]}")

    # [B] Label Smoothing 重建身份：手算展开式 == nn.CrossEntropyLoss(label_smoothing)
    V, eps = 7, 0.1
    logits = torch.randn(1, V)
    target = torch.tensor([3])
    cs = F.cross_entropy(logits, target, label_smoothing=eps)
    # PyTorch 的 label_smoothing：把 eps 均匀分摊在所有 V 类上（含金标），即
    # L = (1-eps)·(-log p[g]) + eps·(1/V)·Σ_c(-log p[c])
    p = F.softmax(logits, -1)
    g = int(target.item())
    manual = (1 - eps) * (-math.log(p[0, g].item())) \
             + eps * (1.0 / V) * sum(-math.log(p[0, c].item()) for c in range(V))
    assert torch.allclose(cs, torch.tensor(manual), atol=1e-5)
    print("[PASS] label-smoothing CE == hand-unrolled identity")

    # [C] 平滑目标分布本身合法：与 PyTorch 同款 eps/V 约定
    # 金标质量 = (1-eps) + eps/V，其余每类 = eps/V → 总质量 = 1
    dist = torch.full((V,), eps / V)
    dist[g] = 1 - eps + eps / V
    assert torch.allclose(dist.sum(), torch.tensor(1.0))
    assert torch.allclose(dist[g], torch.tensor(1 - eps + eps / V)), dist[g]
    print("[PASS] smoothed target sums to 1; gold-class mass = 1-eps+eps/V")

    # [D] 配平滑 + 调度采样的 seq2seq 训练有限且收敛
    torch.manual_seed(0)
    model = Seq2Seq(hid=128).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = build_loader(4000, 64)
    losses = []
    for epoch in range(1, 9):
        tf = scheduled_ratio(epoch, initial=1.0, decay=0.8, minimum=0.3)
        losses.append(run_epoch(model, loader, optim, tf, smoothing=0.1))
    assert all(math.isfinite(x) for x in losses), f"NaN: {losses}"
    assert losses[0] > losses[-1], f"no convergence: {losses}"
    print(f"[PASS] label-smoothed + scheduled-sampling divergence: "
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")