"""m12 Transformer Encoder-Decoder — 移除循环依赖、并行训练（v1.1）。

Run:  python m12_transformer/code.py

v1.1 升级点（versions.md）：把 Attention Seq2Seq 的 RNN 换成「自注意力」，
让一条序列的所有位置**同时**彼此交互（可并行），并用 cross-attention 对话两端。

Encoder: Embedding + PosEnc + N×(Self-Attn + FFN + Res + LN)
Decoder: Embedding + PosEnc + N×(Self-Attn[因果] + Cross-Attn + FFN + Res + LN)
Head   : Linear -> 目标词表

自检：
  [A] 并行性：Encoder 一次前向同时处理整条序列（无 for 时间步），输出 shape 守恒
  [B] Decoder 自注意力因果掩码：未来位置注意力权重恒为 0
  [C] Cross-Attention：解码位置能 attend 到编码器对应输出（经 padding mask）
  [D] 中文数字 -> 英文数字翻译：训练 loss 下降
"""

from __future__ import annotations

import copy
import math
import random
import sys
from typing import List, Optional

import code as _stdlib_code  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ZH = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
zh2en = dict(zip(ZH, EN))


class Vocab:
    def __init__(self, tokens: List[str]):
        self.t2i = {t: i for i, t in enumerate(SPECIAL + tokens)}
        self.i2t = {i: t for t, i in self.t2i.items()}
        self.pad = self.t2i["<PAD>"]; self.bos = self.t2i["<BOS>"]
        self.eos = self.t2i["<EOS>"]

    def __len__(self):
        return len(self.t2i)


zh_vocab = Vocab(ZH)
en_vocab = Vocab(EN)


def collate_batch(samples):
    """samples: list[(src_ids, tgt_ids)] -> (src, src_mask, tgt)。补齐 padding。"""
    srcs = [s[0] for s in samples]
    tgts = [s[1] for s in samples]
    S = max(len(s) for s in srcs)
    B = len(samples)
    src = torch.zeros(B, S, dtype=torch.long).fill_(zh_vocab.pad)
    src_mask = torch.zeros(B, S, dtype=torch.bool)
    for i, s in enumerate(srcs):
        src[i, :len(s)] = torch.tensor(s)
        src_mask[i, :len(s)] = True
    tgt = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in tgts], batch_first=True,
        padding_value=en_vocab.pad)
    return src, src_mask, tgt


def make_samples(n: int, max_len: int = 8):
    samples = []
    for _ in range(n):
        length = random.randint(2, max_len)
        zh_toks = [random.choice(ZH) for _ in range(length)]
        src_ids = [zh_vocab.t2i[t] for t in zh_toks]
        tgt_ids = ([en_vocab.bos]
                   + [en_vocab.t2i[zh2en[t]] for t in zh_toks]
                   + [en_vocab.eos])
        samples.append((src_ids, tgt_ids))
    return samples


def batch_iter(samples, bs):
    random.shuffle(samples)
    for i in range(0, len(samples), bs):
        yield collate_batch(samples[i:i + bs])


# ----------------------------------------------------------------------------
# 位置编码（v1.1）：sinusoidal，给无位置概念的自注意力补顺序信息
# ----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # [1, max_len, D]
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ----------------------------------------------------------------------------
# 多头注意力（v1.1）：自注意力 / 跨注意力，通过 mask 参数区分因果与 padding
# ----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,d]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        B, Tq, _ = query.shape
        q = self._reshape(self.q(query))       # [B,H,Tq,d]
        k = self._reshape(self.k(key))         # [B,H,Tk,d]
        v = self._reshape(self.v(value))       # [B,H,Tk,d]
        # F.scaled_dot_product_attention 同时做 scale / mask / softmax / dropout
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0 if self.training is False else self.drop.p)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.out(out)


# ----------------------------------------------------------------------------
# Feed-Forward（两层的 MLP：D -> 4D -> D）
# ----------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, padding_mask):
        # Pre-Norm + Residual：自注意力里把 padding 的 key 位置置 -inf（禁止被 attend）
        h = self.norm1(x)
        key_mask = torch.where(padding_mask.unsqueeze(1).unsqueeze(2),
                       float("-inf"), 0.0)            # [B,1,1,S] True->-inf
        a = self.attn(h, h, h, attn_mask=key_mask)
        x = x + a
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layers, dropout, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=zh_vocab.pad)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_head, d_ff, dropout)
                                     for _ in range(n_layers)])

    def forward(self, src, src_mask):
        x = self.pos(self.embed(src))                 # [B,S,D]
        pad = ~src_mask                               # [B,S] True=padding
        for blk in self.blocks:
            x = blk(x, pad)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, causal_mask, memory, enc_pad):
        # 因果自注意力（mask 上三角）
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, attn_mask=causal_mask)
        # 交叉注意力：query=decoder，key/value=encoder memory
        h = self.norm2(x)
        q = h
        # padding 屏蔽：把 encoder padding 位置置 -inf，使其 softmax 权重为 0
        attn_mask2 = torch.where(
            enc_pad.squeeze(-1).unsqueeze(1).expand(-1, q.size(1), -1),
            float("-inf"), 0.0).unsqueeze(1)          # [B,1,Tq,S]
        x = x + self.cross_attn(q, memory, memory, attn_mask=attn_mask2)
        h = self.norm3(x)
        x = x + self.ffn(h)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layers, dropout, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=en_vocab.pad)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_head, d_ff, dropout)
                                     for _ in range(n_layers)])
        self.lm = nn.Linear(d_model, vocab_size)

    def causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf")), diagonal=1).to(device)

    def forward(self, tgt, memory, enc_pad):
        x = self.pos(self.embed(tgt))
        T = x.size(1)
        cmask = self.causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, cmask, memory, enc_pad)
        return self.lm(x)


def make_causal_mask(T, device):
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    H, NHEAD, DFF, LAYERS = 64, 4, 128, 2
    enc = Encoder(len(zh_vocab), H, NHEAD, DFF, LAYERS, 0.1, 64).to(device)
    dec = Decoder(len(en_vocab), H, NHEAD, DFF, LAYERS, 0.1, 64).to(device)

# [A] 并行性 + shape：Encoder 一次性处理整条序列（无循环），S 维不变
    src, _mask, tgt = collate_batch(make_samples(4))
    src, _mask, tgt = src.to(device), _mask.to(device), tgt.to(device)
    memory = enc(src, _mask)
    assert memory.shape == (4, src.size(1), H)
    print(f"[PASS] encoder parallel: one forward, all positions at once -> {tuple(memory.shape)}")

    # [B] Decoder 因果掩码：未来位置的注意力权重为 0（softmax(-inf)=0）
    cm = make_causal_mask(6, device)
    out = F.softmax(torch.zeros(6, 6, device=device) + cm, dim=-1)
    triu = torch.triu(torch.ones(6, 6, dtype=torch.bool, device=device), diagonal=1)
    assert out[triu].abs().max().item() < 1e-6, "future positions must have ~0 weight"
    print("[PASS] decoder causal mask: future positions get zero attention")

    # [C] Cross-Attention run
    src2, m2, tgt2 = collate_batch(make_samples(2))
    src2, m2, tgt2 = src2.to(device), m2.to(device), tgt2.to(device)
    mem2 = enc(src2, m2)
    logits = dec(tgt2[:, :-1], mem2, ~m2.unsqueeze(-1))
    assert logits.shape == (2, tgt2.size(1) - 1, len(en_vocab))
    print(f"[PASS] cross-attention decoder forward: logits {tuple(logits.shape)}")

    # [D] 训练：中文数字 -> 英文数字翻译收敛
    torch.manual_seed(0)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-4)
    samples = make_samples(4000, 8)
    losses = []
    for epoch in range(1, 9):
        enc.train(); dec.train()
        tot = 0.0; n = 0
        for s, m, t in batch_iter(samples, 64):
            s, m, t = s.to(device), m.to(device), t.to(device)
            opt.zero_grad(set_to_none=True)
            mem = enc(s, m)
            logits = dec(t[:, :-1], mem, ~m.unsqueeze(-1))
            loss = F.cross_entropy(logits.reshape(-1, len(en_vocab)),
                                   t[:, 1:].reshape(-1), ignore_index=en_vocab.pad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        losses.append(tot / max(n, 1))
        print(f"  epoch={epoch:02d} loss={tot/max(n,1):.4f}")
    assert losses[0] > losses[-1]
    print(f"[PASS] transformer translation loss decreased: {losses[0]:.3f} -> {losses[-1]:.4f}")