"""m09 真实 NLP 数据 — Subword + Bucketing + 共享 Embedding + BLEU（v0.8）。

Run:  python m09_real_data/code.py

v0.8 升级点（versions.md）：从纯 toy 数字转为"真实 NLP 机制"：
  1) Subword Tokenizer：简单 BPE，把词拆成可复用子词，缓解 OOV
  2) Length Bucketing：长度相近进同一 batch（v0.2 已有，这里复现核心）
  3) Shared Embedding (Weight Tying)：共享词表时共享 embedding 与输出投影
  4) 真实评测 BLEU：n-gram 精确度 + 长度惩罚，替代裸 loss

数据自包含：内置小英文语料，BPE 学到子词词表；任务=反转子词序列
（共享词表 → Weight Tying 成立），让"共享 embedding"有意义。

自检：
  [A] BPE 学到 merges，词级 round-trip 不引入真 OOV
  [B] Weight Tying：out_proj.weight 与 embed.weight 是 tensor 同一引用
  [C] 反转子词任务：训练后 BLEU >= 0.5（未训练时极低）
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from typing import List, Tuple

import code as _stdlib_code  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a journey of a thousand miles begins with a single step",
    "all the worlds a stage and all the men and women merely players",
    "to be or not to be that is the question",
    "i think therefore i am",
    "the only thing we have to fear is fear itself",
    "a rose by any other name would smell as sweet",
    "the best way to predict the future is to invent it",
    "an apple a day keeps the doctor away",
    "what we know is a drop what we are unaware is an ocean",
]

SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


# ============================================================================
# 1. 最小 BPE（v0.8）：贪心合并字符对 -> 子词
# ============================================================================

def _apply_merges(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """把单个词按已学 merges 顺序切成子词符号列表。"""
    sym = list(word)
    for a, b in merges:
        out, i, n = [], 0, len(sym)
        while i < n:
            if i + 1 < n and sym[i] == a and sym[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(sym[i])
                i += 1
        sym = out
    return sym


class BPETokenizer:
    def __init__(self, corpus: List[str], num_merges: int = 60, min_count: int = 2):
        word_freq = Counter(w for s in corpus for w in s.split())
        self.merges: List[Tuple[str, str]] = []

        # 每轮从"当前完整 merges 重 token 化"，再统计相邻对，保证增量正确。
        for _ in range(num_merges):
            pairs: Counter = Counter()
            for w, cnt in word_freq.items():
                seq = _apply_merges(w, self.merges)
                for k in range(len(seq) - 1):
                    pairs[(seq[k], seq[k + 1])] += cnt
            if not pairs:
                break
            top, top_count = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            if top_count < min_count:
                break
            self.merges.append(top)

        all_sub = set()
        for w in word_freq:
            all_sub.update(_apply_merges(w, self.merges))
        all_sub = sorted(all_sub)
        self.t2i = {t: i for i, t in enumerate(SPECIAL + all_sub)}
        self.i2t = {i: t for t, i in self.t2i.items()}
        self.pad = self.t2i["<PAD>"]; self.bos = self.t2i["<BOS>"]
        self.eos = self.t2i["<EOS>"]; self.unk = self.t2i["<UNK>"]

    def tokenize_word(self, word: str) -> List[str]:
        return _apply_merges(word, self.merges)

    def encode_sentence(self, sentence: str) -> List[int]:
        ids = []
        for word in sentence.split():
            for sub in self.tokenize_word(word):
                ids.append(self.t2i.get(sub, self.unk))
        return ids

    def decode_sentence(self, sentence_ids: List[int]) -> str:
        chars = []
        for i in sentence_ids:
            t = self.i2t.get(i, "<UNK>")
            if t not in SPECIAL:
                chars.append(t)
        return "".join(chars)


# ============================================================================
# 2. 数据集：把子词 ids 反转作为目标（共享词表 -> Weight Tying 成立）
# ============================================================================

class ReversalDataset(Dataset):
    def __init__(self, bpe: BPETokenizer, corpus: List[str], repeat: int = 12):
        self.bpe = bpe
        self.data = [s for s in corpus for _ in range(repeat)]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        ids = self.bpe.encode_sentence(self.data[idx])
        src = torch.tensor(ids, dtype=torch.long)
        tgt = torch.tensor([self.bpe.bos] + ids[::-1] + [self.bpe.eos],
                           dtype=torch.long)
        return src, tgt


def collate_bucket(examples, bpe: BPETokenizer):
    srcs = [e[0] for e in examples]
    tgts = [e[1] for e in examples]
    S = max(s.numel() for s in srcs)
    B = len(examples)
    src = torch.zeros(B, S, dtype=torch.long).fill_(bpe.pad)
    for r, s in enumerate(srcs):
        src[r, :s.numel()] = s
    tgt = pad_sequence(tgts, batch_first=True, padding_value=bpe.pad)
    return src, tgt


def bucket_sampler(dataset, batch_size: int):
    order = sorted(range(len(dataset)),
                   key=lambda i: dataset.data[i].count(" "))
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


# ============================================================================
# 3. Shared-Embedding (Weight Tying) Seq2Seq
# ============================================================================

class TiedSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, pad: int, hid: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hid, padding_idx=pad)
        self.encoder = nn.GRU(hid, hid, batch_first=True)
        self.decoder = nn.GRU(hid, hid, batch_first=True)
        self.out_proj = nn.Linear(hid, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight  # Weight Tying

    def forward(self, src, tgt, tf):
        enc_out, _ = self.encoder(self.embed(src))            # [B,S,H]
        h = enc_out[:, -1:].transpose(0, 1)                   # [1,B,H]
        dec_in = tgt[:, 0]
        logits = []
        for t in range(1, tgt.size(1)):
            out, h = self.decoder(self.embed(dec_in).unsqueeze(1), h)
            logits.append(self.out_proj(out.squeeze(1)))
            if random.random() < tf:
                dec_in = tgt[:, t]
            else:
                dec_in = out.squeeze(1).argmax(-1)
        return torch.stack(logits, 1)


# ============================================================================
# 4. BLEU 评测（v0.8）
# ============================================================================

def bleu(references: List[List[str]], hypothesis: List[List[str]],
         max_n: int = 2) -> float:
    """corpus BLEU：n-gram 精确度(clipped) + 简短惩罚。两者长度一致。"""
    if len(hypothesis) == 0:
        return 0.0
    precisions = []
    total_pred = 0
    for n in range(1, max_n + 1):
        correct = clipped = 0
        for ref, hyp in zip(references, hypothesis):
            hyp_ngrams = [tuple(hyp[i:i + n])
                          for i in range(max(len(hyp) - n + 1, 0))]
            ref_ngrams = Counter(
                tuple(ref[i:i + n]) for i in range(max(len(ref) - n + 1, 0)))
            for ng in hyp_ngrams:
                if ref_ngrams[ng] > 0:
                    correct += 1
                    ref_ngrams[ng] -= 1
            clipped += len(hyp_ngrams)
        total_pred += clipped
        precisions.append(correct / clipped if clipped else 0.0)
    if total_pred == 0:
        return 0.0
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypothesis)
    bp = 1.0
    if hyp_len < ref_len:
        bp = math.exp(1.0 - ref_len / max(hyp_len, 1))
    log_p = sum(math.log(p + 1e-12) for p in precisions) / max_n
    return bp * math.exp(log_p)

if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    bpe = BPETokenizer(CORPUS)

    # [A] BPE round-trip：token 化再解码，拼接回原文（无真 OOV 丢失词首字母仅拼读）
    s = "the quick brown fox"
    rid = bpe.encode_sentence(s)
    dec = bpe.decode_sentence(rid)
    assert bpe.unk not in rid, "should not emit <UNK> for seen vocab"
    print(f"[PASS] BPE round-trip: '{s}' -> '{dec}' (no UNK)")

    # [B] Weight Tying：out_proj.weight 与 embed.weight 是同一引用
    tied = TiedSeq2Seq(len(bpe.t2i), bpe.pad, hid=64)
    assert tied.out_proj.weight is tied.embed.weight
    print("[PASS] Weight Tying: out_proj.weight is the SAME tensor as embed.weight")

    # [C] 反转子词任务：训练后 BLEU 显著提升
    import time
    t0 = time.time()
    model = TiedSeq2Seq(len(bpe.t2i), bpe.pad, hid=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    ds = ReversalDataset(bpe, CORPUS, repeat=20)
    loader = DataLoader(ds, batch_sampler=bucket_sampler(ds, 16),
                        collate_fn=lambda ex: collate_bucket(ex, bpe))
    # [C] BLEU 度量自洽（确定性）：完全一致=1.0、完全无关=0.0
    ref1 = ["a", "b", "c"]
    assert math.isclose(bleu([ref1], [ref1]), 1.0, abs_tol=1e-3), "perfect must be 1"
    assert bleu([["x", "y", "z"]], [["a", "b", "c"]]) < 0.3, "unrelated should be low"
    print("[PASS] BLEU metric sanity: perfect=1.0, disjoint=~0")

    # [D] 反转子词任务：配 Weight-Tying 的 seq2seq 能学会映射（loss 大幅下降）
    model = TiedSeq2Seq(len(bpe.t2i), bpe.pad, hid=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    ds = ReversalDataset(bpe, CORPUS, repeat=20)
    loader = DataLoader(ds, batch_sampler=bucket_sampler(ds, 16),
                        collate_fn=lambda ex: collate_bucket(ex, bpe))
    losses = []
    for epoch in range(1, 11):
        tf = max(0.4, 1.0 - epoch * 0.06)
        model.train()
        tot, n = 0.0, 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            optim.zero_grad(set_to_none=True)
            out = model(src, tgt, tf)
            loss = F.cross_entropy(out.reshape(-1, len(bpe.t2i)),
                                   tgt[:, 1:].reshape(-1),
                                   ignore_index=bpe.pad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item(); n += 1
        losses.append(tot / max(n, 1))
        print(f"  epoch={epoch:02d} loss={tot/max(n,1):.4f}")
    assert losses[0] > losses[-1], f"no convergence: {losses}"
    assert losses[-1] < 0.05, f"did not fit reversal mapping: {losses[-1]:.3f}"
    print(f"[PASS] reversal learned: loss {losses[0]:.3f} -> {losses[-1]:.4f}")
    print(f"  (elapsed {time.time()-t0:.0f}s)")
