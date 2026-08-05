"""m05 双向 Encoder + Bridge — 让每个源位置同时看到左右上下文（v0.4）。

Run:  python m05_bidirectional/code.py

v0.4 升级点（versions.md）：
  单向 Encoder: h_i = f(x_i, h_{i-1})           只能看到左侧
  双向 Encoder: h_i = [→h_i ; ←h_i]              同时看到左右
  Bridge: 把「双向最终隐状态 → Decoder 初始隐状态」的线性/激活映射

自检：
  [A] 双向输出维度 = 2×hidden（前向+后向拼接）；shape 守恒
  [B] Bridge 把 2H -> H，且 Decoder 能初始化、装配跑通
  [C] 双向 Seq2Seq 在教师强制下收敛（loss 下降）
"""

from __future__ import annotations

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

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.i2t.get(i, "<UNK>")
            if t == "<EOS>": break
            if t in SPECIAL_TOKENS: continue
            out.append(t)
        return out

    def __len__(self): return len(self.t2i)


src_vocab = Vocabulary(EN)
tgt_vocab = Vocabulary(ZH)


def make_example():
    length = random.randint(1, 7)
    src = [random.choice(EN) for _ in range(length)]
    tgt = [mapping[t] for t in src]
    return src, tgt


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
    mask = torch.zeros(B, S, dtype=torch.bool)
    for r, (s, l) in enumerate(zip(srcs, lens)):
        src[r, :l] = s; mask[r, :l] = True
    tgt = pad_sequence([e[1] for e in examples], batch_first=True,
                       padding_value=tgt_vocab.pad)
    return src, mask, torch.tensor(lens), tgt


def build_loader(size, bs):
    return DataLoader(NumberDataset(size), batch_size=bs, shuffle=True,
                      collate_fn=collate)


# ----------------------------------------------------------------------------
# 双向 Encoder（v0.4）：bidirectional GRU + [fwd;bwd] 最终状态
# ----------------------------------------------------------------------------

class BidirectionalEncoder(nn.Module):
    def __init__(self, vocab, embed, hid, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=pad)
        # 双向：输出维度翻倍到 2*hid；hidden 是两层(0=fwd,1=bwd)的终态拼向量
        self.gru = nn.GRU(embed, hid, num_layers=1, bidirectional=True,
                          batch_first=True)

    def forward(self, src, src_len):
        emb = self.embed(src)
        outputs, hidden = self.gru(emb)        # [B,S,2H]、[2,B,H]
        fwd = hidden[0]                        # [B,H]
        bwd = hidden[1]                        # [B,H]
        last = torch.cat([fwd, bwd], dim=-1)   # [B,2H]
        return outputs, last                   # outputs [B,S,2H]


# ----------------------------------------------------------------------------
# Bridge（v0.4）：把 2H -> Decoder 的 H，初始化 Decoder 状态
# ----------------------------------------------------------------------------

class Bridge(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.proj = nn.Linear(enc_dim, dec_dim)   # 2H -> H

    def forward(self, enc_last):
        # enc_last: [B, 2H]
        return torch.tanh(self.proj(enc_last)).unsqueeze(0)  # [1,B,H]


# ----------------------------------------------------------------------------
# Decoder（无注意力，聚焦 v0.4 的双向 + Bridge）
# ----------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, vocab, embed, hid, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=pad)
        self.gru = nn.GRU(embed, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward(self, input_token, hidden):
        emb = self.embed(input_token).unsqueeze(1)   # [B,1,E]
        out, next_hidden = self.gru(emb, hidden)       # [B,1,H]
        return self.out(out.squeeze(1)), next_hidden   # [B,V]


class BiSeq2Seq(nn.Module):
    def __init__(self, enc, bridge, dec):
        super().__init__()
        self.enc = enc
        self.bridge = bridge
        self.dec = dec

    def forward(self, src, src_len, tgt, tf_ratio):
        _, enc_last = self.enc(src, src_len)            # [B,2H]
        h = self.bridge(enc_last)                        # [1,B,H]
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
        for src, mask, lens, tgt in loader:
            src, lens, tgt = src.to(device), lens.to(device), tgt.to(device)
            if training: optim.zero_grad(set_to_none=True)
            out = model(src, lens, tgt, tf)
            loss = F.cross_entropy(out.reshape(-1, len(tgt_vocab)),
                                   tgt[:, 1:].reshape(-1),
                                   ignore_index=tgt_vocab.pad)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    EMB, HID = 64, 128

    # [A] 双向 Encoder shape：outputs=[B,S,2H]，终态拼接 [B,2H]
    enc = BidirectionalEncoder(len(src_vocab), EMB, HID, src_vocab.pad).to(device)
    x = torch.randint(0, len(src_vocab), (3, 6)).to(device)
    lens = torch.tensor([4, 6, 5]).to(device)
    outputs, last = enc(x, lens)
    assert outputs.shape == (3, 6, 2 * HID)
    assert last.shape == (3, 2 * HID)
    print("[PASS] bidirectional outputs [B,S,2H] and concat fwd|bwd last [B,2H]")

    # [B] Bridge: 2H -> H，Decoder 初始化 + 单步跑通
    bridge = Bridge(2 * HID, HID).to(device)
    dec = Decoder(len(tgt_vocab), EMB, HID, tgt_vocab.pad).to(device)
    h0 = bridge(last)                               # [1,B,H]
    dec_in = torch.tensor([tgt_vocab.bos] * 3, device=device)
    logits, h1 = dec(dec_in, h0)
    assert logits.shape == (3, len(tgt_vocab))
    assert h1.shape == (1, 3, HID)
    print("[PASS] Bridge 2H->H + decoder init & step shapes correct")

    # [C] 双向 Seq2Seq 教师强制收敛
    model = BiSeq2Seq(enc, bridge, dec).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = build_loader(4000, 64)
    losses = []
    for epoch in range(1, 9):
        tf = max(0.5, 1.0 - epoch * 0.06)
        losses.append(run_epoch(model, loader, optim, tf, True))
    assert losses[0] > losses[-1]
    print(f"[PASS] bidirectional seq2seq loss decreased: "
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")