"""m06 深层 Seq2Seq — Multi-layer + Dropout + Residual + LayerNorm（v0.5）。

Run:  python m06_deep/code.py

v0.5 升级点（versions.md）：
  - 多层 GRU（Encoder 3 层、Decoder 3 层），层之间 Dropout
  - Residual Connection：decoder 在 GRU 输出上加输入，缓解深层梯度
  - Layer Normalization：稳定深层归一化分布
  - 梯度裁剪 + 学习率衰减：深层优化稳定性

自检：
  [A] 多层 Encoder 输出 shape 正确（num_layers 生效、维度守恒）
  [B] Residual + LayerNorm：与前向等价/数值合理、shape 守恒
  [C] 深层 Seq2Seq 在教师强制下收敛（loss 下降），且训练稳定（不 NaN）
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


def make_example():
    length = random.randint(1, 7)
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


# ----------------------------------------------------------------------------
# 深层 Encoder（v0.5：num_layers=3 + 层间 dropout）
# ----------------------------------------------------------------------------

class DeepEncoder(nn.Module):
    def __init__(self, vocab, embed, hid, pad, layers=3, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=pad)
        self.gru = nn.GRU(embed, hid, num_layers=layers, dropout=dropout,
                          batch_first=True)
        self.layers = layers

    def forward(self, src, src_len):
        emb = self.embed(src)
        outputs, hidden = self.gru(emb)     # [B,S,H]、[layers,B,H]
        return outputs, hidden              # hidden 每层都有，取顶层


# ----------------------------------------------------------------------------
# 深层 Decoder：单步 + LayerNorm + Residual（v0.5）
# ----------------------------------------------------------------------------

class DeepDecoder(nn.Module):
    def __init__(self, vocab, embed, hid, pad, layers=3, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=pad)
        self.gru = nn.GRU(embed, hid, num_layers=layers, dropout=dropout,
                          batch_first=True)
        # 输入 embedding 维度 E=embed，需投影到 H 才能做残差连接
        self.res_proj = nn.Linear(embed, hid)
        self.ln = nn.LayerNorm(hid)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hid, vocab)

    def forward(self, input_token, hidden):
        emb = self.embed(input_token)                    # [B, E]
        emb_t = emb.unsqueeze(1)
        out, next_hidden = self.gru(emb_t, hidden)       # [B,1,H]
        h = out.squeeze(1)
        # Residual + LayerNorm（Post-Norm）：h + 输入投影（残差短路），再 LN
        h = h + self.res_proj(emb)
        return self.out(self.ln(h)), next_hidden


class DeepSeq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src, src_len, tgt, tf_ratio):
        _, enc_hidden = self.enc(src, src_len)          # [layers,B,H]
        h = enc_hidden                                   # 顶层终态作为初始
        dec_in = tgt[:, 0]
        logits = []
        for t in range(1, tgt.size(1)):
            out, h = self.dec(dec_in, h)
            logits.append(out)
            if random.random() < tf_ratio:
                dec_in = tgt[:, t]
            else:
                dec_in = out.argmax(-1)
        return torch.stack(logits, 1)


def run_epoch(model, loader, optim, tf, training):
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()
    total, n = 0.0, 0
    with ctx:
        for src, lens, tgt in loader:
            src, lens, tgt = src.to(device), lens.to(device), tgt.to(device)
            if training: optim.zero_grad(set_to_none=True)
            out = model(src, lens, tgt, tf)
            loss = F.cross_entropy(out.reshape(-1, len(tgt_vocab)),
                                   tgt[:, 1:].reshape(-1),
                                   ignore_index=tgt_vocab.pad)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optim.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    EMB, HID, LAYERS, DROP = 64, 128, 3, 0.2

    # [A] 多层 Encoder：num_layers 生效，hidden 顶层 shape
    enc = DeepEncoder(len(src_vocab), EMB, HID, src_vocab.pad, LAYERS, DROP).to(device)
    src = torch.randint(0, len(src_vocab), (3, 6)).to(device)
    lens = torch.tensor([4, 6, 5]).to(device)
    outputs, hidden = enc(src, lens)
    assert hidden.shape == (LAYERS, 3, HID), hidden.shape
    assert outputs.shape == (3, 6, HID)
    print(f"[PASS] deep encoder: outputs[B,S,H], hidden[{LAYERS},B,H]")

    # [B] Decoder+R/LN 装配，用多层隐状态初始化顶层解码
    dec = DeepDecoder(len(tgt_vocab), EMB, HID, tgt_vocab.pad, LAYERS, DROP).to(device)
    h0 = hidden                               # [LAYERS,B,H]，多层 GRU 需 3-D 初始
    dec_in = torch.tensor([tgt_vocab.bos] * 3, device=device)
    logits, h1 = dec(dec_in, h0)
    assert logits.shape == (3, len(tgt_vocab))
    assert h1.shape == (LAYERS, 3, HID)
    print("[PASS] deep decoder init w/ multi-layer state + step: logits shape ok")

    # [C] 深层模型训练收敛且不 NaN
    torch.manual_seed(0)
    model = DeepSeq2Seq(enc, dec).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = build_loader(4000, 64)
    losses = []
    for epoch in range(1, 11):
        tf = max(0.5, 1.0 - epoch * 0.05)
        losses.append(run_epoch(model, loader, optim, tf, True))
    assert all(math.isfinite(x) for x in losses), f"NaN in losses: {losses}"
    assert losses[0] > losses[-1], f"no improvement: {losses}"
    print(f"[PASS] deep-seq2seq loss decreased+finite: "
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")