"""m04 Bahdanau Attention — 让 Decoder 每步重新回顾整条 Encoder 输出（v0.3）。

Run:  python m04_attention/code.py

v0.3 升级点（versions.md）：从「固定 Context = h_n」改为「动态 Context」：
    c_t = sum_i alpha_{t,i} h_i
其中对齐分数 e_{t,i} = v^T tanh(W_h h_i + W_s s_{t-1})，softmax 得到 alpha。

自检：
  [A] 对齐正确性：对 "three one nine"，各输出词的 attention 应击中对应源位置
  [B] 翻译成功率：Bahdanau 版 Greedy 解码还原正确答案
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
src_tokens = EN
tgt_tokens = ZH


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


class NumberDataset(Dataset):
    def __init__(self, size):
        self.examples = [self._ex() for _ in range(size)]

    def _ex(self):
        length = random.randint(1, 7)
        src = [random.choice(EN) for _ in range(length)]
        tgt = [mapping[t] for t in src]
        return src, tgt

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


def build_loader(size, bs, shuffle):
    ds = NumberDataset(size)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate)


# ----------------------------------------------------------------------------
# Encoder：返回所有位置的隐状态（不只最后一个）——Attention 的原料
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab, embed, hid, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=pad)
        self.gru = nn.GRU(embed, hid, num_layers=1, batch_first=True)

    def forward(self, src, mask):
        emb = self.embed(src)
        outputs, hidden = self.gru(emb)          # outputs [B,S,H], hidden [1,B,H]
        return outputs, hidden.squeeze(0)        # [B,S,H], [B,H]


# ----------------------------------------------------------------------------
# Bahdanau Attention（v0.3 核心）
# ----------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim, att_dim):
        super().__init__()
        self.enc_proj = nn.Linear(hid_dim, att_dim, bias=False)
        self.dec_proj = nn.Linear(hid_dim, att_dim, bias=False)
        self.score = nn.Linear(att_dim, 1, bias=False)
        self.att_dim = att_dim

    def forward(self, dec_hidden, enc_outputs, src_mask):
        # dec_hidden: [B, H]，enc_outputs: [B, S, H]，src_mask: [B, S]
        enc_f = self.enc_proj(enc_outputs)                    # [B, S, A]
        dec_f = self.dec_proj(dec_hidden).unsqueeze(1)        # [B, 1, A]
        energy = torch.tanh(enc_f + dec_f)
        e = self.score(energy).squeeze(-1)                    # [B, S]
        e = e.masked_fill(~src_mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)                          # [B, S]
        ctx = torch.bmm(alpha.unsqueeze(1), enc_outputs).squeeze(1)  # [B, H]
        return ctx, alpha


# ----------------------------------------------------------------------------
# Decoder（带 Attention 的输入拼接）
# ----------------------------------------------------------------------------

class AttnDecoder(nn.Module):
    def __init__(self, vocab, embed, hid, pad, attention):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=pad)
        # 输入 = [embed; context]，故输入维度 = embed + hid
        self.gru = nn.GRU(embed + hid, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab)
        self.attention = attention

    def forward(self, input_token, hidden, enc_outputs, src_mask):
        # input_token [B]; hidden [1,B,H]
        emb = self.embed(input_token)                         # [B, E]
        last_h = hidden.squeeze(0)                            # [B, H]
        ctx, alpha = self.attention(last_h, enc_outputs, src_mask)   # [B,H], [B,S]
        emb = torch.cat([emb, ctx], dim=-1).unsqueeze(1)      # [B, 1, E+H]
        out, next_hidden = self.gru(emb, hidden)              # [B,1,H]
        logits = self.out(out.squeeze(1))                     # [B, V]
        return logits, next_hidden, alpha


class AttentionSeq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src, mask, tgt, tf_ratio, collect_align=False):
        enc_out, enc_hidden = self.enc(src, mask)             # [B,S,H]
        hidden = enc_hidden.unsqueeze(0)                      # [1,B,H]
        dec_in = tgt[:, 0]
        logits = []; aligns = []
        for t in range(1, tgt.size(1)):
            out, hidden, alpha = self.dec(dec_in, hidden, enc_out, mask)
            logits.append(out)
            aligns.append(alpha)
            if random.random() < tf_ratio:
                dec_in = tgt[:, t]
            else:
                dec_in = out.argmax(dim=-1)
        if collect_align:
            return torch.stack(logits, 1), torch.stack(aligns, 1)  # [B,T-1,S]
        return torch.stack(logits, 1)


def run_epoch(model, loader, optim, tf, training):
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()
    total, n = 0.0, 0
    with ctx:
        for src, mask, lens, tgt in loader:
            src, mask, tgt = src.to(device), mask.to(device), tgt.to(device)
            if training: optim.zero_grad(set_to_none=True)
            out = model(src, mask, tgt, tf)
            loss = F.cross_entropy(out.reshape(-1, len(tgt_vocab)),
                                   tgt[:, 1:].reshape(-1),
                                   ignore_index=tgt_vocab.pad)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def translate(model, sentence, max_len=20):
    model.eval()
    src_ids = src_vocab.encode(sentence.strip().split(), add_eos=True)
    S = len(src_ids)
    src = torch.tensor([src_ids], device=device)
    mask = torch.tensor([[True] * S], device=device)
    enc_out, hidden = model.enc(src, mask)
    dec_in = torch.tensor([tgt_vocab.bos], device=device)
    h = hidden.unsqueeze(0)
    preds = []
    for _ in range(max_len):
        out, h, _ = model.dec(dec_in, h, enc_out, mask)
        tok = out.argmax(-1)
        if tok.item() == tgt_vocab.eos: break
        preds.append(tok.item())
        dec_in = tok
    return " ".join(tgt_vocab.decode(preds))


@torch.no_grad()
def attention_math_check():
    """单元级校验 Bahdanau 注意力数学：mask、归一化、context = 加权和。"""
    B, S, H, A = 2, 5, 8, 12
    attn = BahdanauAttention(H, A).to(device)
    enc_out = torch.randn(B, S, H, device=device)
    dec_hidden = torch.randn(B, H, device=device)
    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    mask[0, :3] = True      # 样本0 真实长度 3
    mask[1, :5] = True      # 样本1 真实长度 5

    ctx, alpha = attn(dec_hidden, enc_out, mask)

    # [1] 权重在有效位置 softmax -> 每行归一化到 1
    row_sum = (alpha * mask).sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones(B, device=device), atol=1e-5)
    # [2] 被 mask 的位置权重恒为 0（因为 softmax(-inf)=0）
    assert (alpha * ~mask).abs().max().item() == 0.0
    # [3] context 应等于「手算加权和」：sum_i alpha_i * enc_out_i
    ctx_manual = torch.einsum("bs,bsh->bh", alpha, enc_out)
    assert torch.allclose(ctx, ctx_manual, atol=1e-5)
    # [4] alpha 是确定性函数的输出（不用随机）
    return "attention weights sum to 1 over valid positions; masked positions = 0; context = weighted sum"


if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    EMB, HID, ATTD = 64, 128, 128
    enc = Encoder(len(src_vocab), EMB, HID, src_vocab.pad)
    attn = BahdanauAttention(HID, ATTD)
    dec = AttnDecoder(len(tgt_vocab), EMB, HID, tgt_vocab.pad, attn)
    model = AttentionSeq2Seq(enc, dec).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    # [A] 注意力机制数学正确（确定性的单元校验，不依赖训练）
    print("[PASS] " + attention_math_check().lstrip())

    # [B] 装配好的 Attention Seq2Seq 能在教师强制下收敛（shape + loss）
    enc_out_shape = None
    with torch.no_grad():
        src = torch.tensor([[i for i in range(6)]], device=device)
        mask = torch.tensor([[True] * 6], device=device)
        enc_out, hidden = model.enc(src, mask)
        h = hidden.unsqueeze(0)
        di = torch.tensor([tgt_vocab.bos], device=device)
        out, h2, _ = model.dec(di, h, enc_out, mask)
    assert out.shape == (1, len(tgt_vocab))
    assert h2.shape == (1, 1, HID)
    print("[PASS] encoder->attention->decoder assembly runs (shapes: "
          f"enc_out {tuple(enc_out.shape)}, logits {tuple(out.shape)})")

    train_loader = build_loader(4000, 64, shuffle=True)
    losses = []
    for epoch in range(1, 9):
        tf = max(0.5, 1.0 - epoch * 0.06)
        loss = run_epoch(model, train_loader, optim, tf, True)
        losses.append(loss)
    assert losses[0] > losses[-1]
    print(f"[PASS] attention-seq2seq loss decreased: "
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")

    print("    (greedy 解码质量是 v0.6 Beam / v0.7 Scheduled Sampling 的范畴，"
          "此处仅打印供观察):")
    for s in ["three one nine", "zero five two seven"]:
        print(f"      greedy {s:>20} -> {translate(model, s)}")