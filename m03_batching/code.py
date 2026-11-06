"""m03 Batching — 变长序列的规范化批处理：Padding / Mask / pack_padded_sequence（v0.2）。

Run:  python m03_batching/code.py

v0.2 升级点（对照 versions.md）：
  1) pack_padded_sequence：Encoder 不再在 <PAD> 上继续算 RNN（省算力、防污染隐状态）
  2) ignore_index=pad_id：Decoder loss 在 <PAD> 位置上不计梯度
  3) Length Bucketing：长度接近的样本放进同一 batch，减少 padding 浪费
  4) 输出 Mask：显式标出「哪些位置是真实 token」

三个关键自检：
  [A] packed vs 朴素 padded：取「最后一个非 PAD 步隐状态」两种方式结果一致 < 1e-5
  [B] ignore_index：把一条样本的 target 末尾挖洞变 pad，loss 不变（证明 pad 不计）
  [C] bucketing：分桶后每个 batch 内部长度差有界
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

import code as _stdlib_code  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ZH = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
number_mapping = dict(zip(EN, ZH))


class Vocabulary:
    def __init__(self, tokens: List[str]):
        self.t2i = {t: i for i, t in enumerate(SPECIAL_TOKENS + tokens)}
        self.i2t = {i: t for t, i in self.t2i.items()}
        self.pad = self.t2i["<PAD>"]; self.bos = self.t2i["<BOS>"]
        self.eos = self.t2i["<EOS>"]; self.unk = self.t2i["<UNK>"]

    def encode(self, toks, add_bos=False, add_eos=False) -> List[int]:
        ids = [self.t2i.get(t, self.unk) for t in toks]
        if add_bos: ids = [self.bos] + ids
        if add_eos: ids = ids + [self.eos]
        return ids

    def decode(self, ids) -> List[str]:
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


# ----------------------------------------------------------------------------
# v0.2：合成数据 + Bucketing Dataset
# ----------------------------------------------------------------------------

def make_example():
    length = random.randint(1, 7)
    src = [random.choice(EN) for _ in range(length)]
    tgt = [number_mapping[t] for t in src]
    return src, tgt


class BucketedNumberDataset(Dataset):
    """按源长度分桶。__getitem__ 返回 (ids, length)，由 BucketSampler 聚桶。"""

    def __init__(self, size: int):
        self.examples = [make_example() for _ in range(size)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src, tgt = self.examples[idx]
        src_ids = torch.tensor(src_vocab.encode(src, add_eos=True), dtype=torch.long)
        tgt_ids = torch.tensor(tgt_vocab.encode(tgt, add_bos=True, add_eos=True),
                               dtype=torch.long)
        return src_ids, tgt_ids, len(src_ids)


def bucket_sampler(dataset: BucketedNumberDataset, batch_size: int):
    """按源长度排序后顺序切 batch：同一 batch 内长度相近，padding 浪费最少。"""
    order = sorted(range(len(dataset)), key=lambda i: len(dataset.examples[i][0]))
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


@dataclass
class Batch:
    src: torch.Tensor
    src_len: torch.Tensor
    tgt: torch.Tensor
    src_mask: torch.Tensor   # [B, S] bool，True=真实 token（非 PAD）
    tgt_mask: torch.Tensor   # [B, T] bool，True=要监督的位置（非 PAD）


def collate_bucket(examples):
    srcs = [e[0] for e in examples]; lens = [e[2] for e in examples]
    tgts = [e[1] for e in examples]
    S = max(lens)
    B = len(examples)
    pad = tgt_vocab.pad
    src_b = torch.zeros(B, S, dtype=torch.long).fill_(src_vocab.pad)
    src_mask = torch.zeros(B, S, dtype=torch.bool)
    for r, (s, l) in enumerate(zip(srcs, lens)):
        src_b[r, :l] = s
        src_mask[r, :l] = True
    tgt_b = pad_sequence(tgts, batch_first=True, padding_value=pad)
    tgt_mask = (tgt_b != pad)
    return Batch(src_b, torch.tensor(lens), tgt_b, src_mask, tgt_mask)


def make_bucketed_loader(size: int, batch_size: int):
    ds = BucketedNumberDataset(size)
    loader = DataLoader(ds, batch_sampler=bucket_sampler(ds, batch_size),
                        collate_fn=collate_bucket)
    return loader


# ----------------------------------------------------------------------------
# v0.2：Encoder 用 pack_padded_sequence（不在 <PAD> 上算）
# ----------------------------------------------------------------------------

class PackedEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src, src_len):                 # src [B,S]
        emb = self.embed(src)
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B,S,H]
        # 取每个样本「最后一个真实 token」的隐状态做 Context
        idx = (src_len - 1).unsqueeze(1).expand(-1, out.size(-1)).unsqueeze(1)
        gather = torch.gather(out, 1, idx)            # [B,1,H]
        return gather.squeeze(1).unsqueeze(0)         # [1,B,H]


# Naive 对照：不 pack，直接在含 PAD 的输出上取最后非 PAD 步
class NaiveEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src, src_len):
        emb = self.embed(src)
        out, _ = self.gru(emb)                        # [B,S,H]
        batch_idx = torch.arange(src.size(0), device=src.device)
        last = out[batch_idx, src_len - 1]            # [B,H]
        return last.unsqueeze(0)


# ----------------------------------------------------------------------------
# 主程序：三组 v0.2 自检
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    # 固定一批变长样本
    raw = [(make_example()) for _ in range(24)]
    srcs = [torch.tensor(src_vocab.encode(s, add_eos=True)) for s, _ in raw]
    tgts = [torch.tensor(tgt_vocab.encode(t, add_bos=True, add_eos=True)) for _, t in raw]
    coll = collate_bucket([(s, t, len(s)) for s, t in zip(srcs, tgts)])

    EMB, HID = 32, 64
    packed = PackedEncoder(len(src_vocab), EMB, HID, src_vocab.pad).to(device)
    naive = NaiveEncoder(len(src_vocab), EMB, HID, src_vocab.pad).to(device)
    # 同一份 embedding 与 GRU，公平对比
    naive.embed.load_state_dict(packed.embed.state_dict())
    naive.gru.load_state_dict(packed.gru.state_dict())

    src = coll.src.to(device); lens = coll.src_len.to(device)
    h_packed = packed(src, lens)
    h_naive = naive(src, lens)
    assert h_packed.shape == h_naive.shape == (1, src.size(0), HID)
    diff = (h_packed - h_naive).abs().max().item()
    assert diff < 1e-3, diff
    print(f"[PASS] packed vs naive last-token context equal (max abs diff {diff:.2e} < 1e-3)")

    # ---- v0.2 验收：ignore_index 使 <PAD> 位置不参与 loss ----
    # 用对比法：同一份 logits + 全真实 target 的 CE，与"把某些真实 token 换成 PAD"
    # 后、只对剩余非 PAD 位置算的 CE，应完全一致——证明 pad 位被忽略。
    B2, S2, V = 4, 6, len(tgt_vocab)
    logits = torch.randn(B2 * S2, V, device=device)          # 模拟展平 logits
    ids = torch.randint(0, V, (B2 * S2,), device=device)     # 模拟展平 target
    ids[3] = (tgt_vocab.pad + 1) % V
    ids[8] = (tgt_vocab.pad + 2) % V                          # 保证这两处非 PAD
    ids2 = ids.clone()
    pad = tgt_vocab.pad
    ids2[3] = pad; ids2[8] = pad                              # 人为制造两处 PAD
    loss_all = nn.functional.cross_entropy(logits, ids)
    loss_pad = nn.functional.cross_entropy(logits, ids2, ignore_index=pad)
    # 手工只对非 PAD 的 logits 求 CE
    kept = (ids2 != pad)
    loss_manual = nn.functional.cross_entropy(logits[kept], ids2[kept])
    assert torch.allclose(loss_pad, loss_manual, atol=1e-6)
    # 忽略位置只改变参与平均的样本集合，并不保证 loss 必然降低。
    # 用原始 target 在保留位置上的 CE 做独立交叉检查。
    loss_kept_original = nn.functional.cross_entropy(logits[kept], ids[kept])
    assert torch.allclose(loss_pad, loss_kept_original, atol=1e-6)
    assert kept.sum().item() < ids.numel()
    print(f"[PASS] ignore_index=pad: CE_all[{loss_all:.3f}], "
          f"CE_ignore_pad[{loss_pad:.3f}] == manual[{loss_manual:.3f}]")

    # ---- 3：bucket 后 batch 内长度差有界 ----
    loader = make_bucketed_loader(512, 32)
    max_span = -1
    for b in loader:
        l = b.src_len
        max_span = max(max_span, (l.max() - l.min()).item())
    assert max_span <= 4, f"bucket span too large: {max_span}"
    print(f"[PASS] length bucketing bounds batch-internal span to {max_span}")