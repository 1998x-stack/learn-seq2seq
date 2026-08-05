"""m02 Seq2Seq MVP — GRU Encoder-Decoder，英文数字序列->中文数字序列（v0.1）。

Run:  python m02_seq2seq_mvp/code.py

对应 versions.md 第五节 MVP 与迭代 v0.1：最小可运行 Encoder-Decoder 闭环。
核心假设 c = h_n（整条输入压成一个固定向量），包含：
  - 独立输入/输出词表 + <PAD>/<BOS>/<EOS>/<UNK>
  - 变长序列 + Padding + collate
  - Encoder（GRU 读入，取最后隐状态）
  - Decoder（GRU 逐词生成）
  - Teacher Forcing + Cross Entropy + 梯度裁剪 + Greedy Decoding
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

# 防止 `import code` 被本文件遮蔽
import code as _stdlib_code  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 1. 词表（v0.1）
# ----------------------------------------------------------------------------

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
ENGLISH_NUMBERS = ["zero", "one", "two", "three", "four",
                   "five", "six", "seven", "eight", "nine"]
CHINESE_NUMBERS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]


class Vocabulary:
    def __init__(self, tokens: List[str]):
        all_tokens = SPECIAL_TOKENS + tokens
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

    def encode(self, tokens: List[str], add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, remove_special: bool = True) -> List[str]:
        out = []
        for i in ids:
            t = self.id_to_token.get(i, "<UNK>")
            if t == "<EOS>":
                break
            if remove_special and t in SPECIAL_TOKENS:
                continue
            out.append(t)
        return out

    def __len__(self) -> int:
        return len(self.token_to_id)


src_vocab = Vocabulary(ENGLISH_NUMBERS)
tgt_vocab = Vocabulary(CHINESE_NUMBERS)
number_mapping = dict(zip(ENGLISH_NUMBERS, CHINESE_NUMBERS))


# ----------------------------------------------------------------------------
# 2. 合成数据集（v0.1）
# ----------------------------------------------------------------------------

def generate_example(min_len: int = 1, max_len: int = 7) -> Tuple[List[str], List[str]]:
    length = random.randint(min_len, max_len)
    src = [random.choice(ENGLISH_NUMBERS) for _ in range(length)]
    tgt = [number_mapping[t] for t in src]
    return src, tgt


class NumberDataset(Dataset):
    def __init__(self, size: int):
        self.examples = [generate_example() for _ in range(size)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.examples[idx]
        # Encoder 不需要 BOS，但加 EOS；Decoder 需 BOS + EOS。
        src_ids = src_vocab.encode(src_tokens, add_bos=False, add_eos=True)
        tgt_ids = tgt_vocab.encode(tgt_tokens, add_bos=True, add_eos=True)
        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long))


@dataclass
class Batch:
    source: torch.Tensor
    source_lengths: torch.Tensor
    target: torch.Tensor


def collate_batch(examples):
    srcs, tgts = map(list, zip(*examples))
    lengths = torch.tensor([s.numel() for s in srcs], dtype=torch.long)
    src_batch = pad_sequence(srcs, batch_first=True, padding_value=src_vocab.pad_id)
    tgt_batch = pad_sequence(tgts, batch_first=True, padding_value=tgt_vocab.pad_id)
    return Batch(source=src_batch, source_lengths=lengths, target=tgt_batch)


def build_loader(size: int, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(NumberDataset(size), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch)


# ----------------------------------------------------------------------------
# 3. Encoder（v0.1：取最终隐状态作 Context）
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, source, source_lengths):
        embedded = self.embedding(source)          # [B, S, E]
        outputs, _ = self.gru(embedded)            # [B, S, H]
        # 取每个样本最后一个非 PAD 时间步的隐状态
        last_idx = (source_lengths - 1).to(source.device)
        batch_idx = torch.arange(source.size(0), device=source.device)
        last_hidden = outputs[batch_idx, last_idx]  # [B, H]
        return last_hidden.unsqueeze(0)             # [1, B, H]（num_layers x B x H）


# ----------------------------------------------------------------------------
# 4. Decoder（v0.1：逐词 GRU）
# ----------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden):
        # input_token: [B]，hidden: [1, B, H]
        emb = self.embedding(input_token).unsqueeze(1)   # [B, 1, E]
        out, next_hidden = self.gru(emb, hidden)           # [B, 1, H]
        logits = self.out(out.squeeze(1))                  # [B, V]
        return logits, next_hidden


# ----------------------------------------------------------------------------
# 5. Seq2Seq (v0.1: Encoder + Decoder + Teacher Forcing)
# ----------------------------------------------------------------------------

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, source_lengths, target, teacher_forcing_ratio=0.5):
        hidden = self.encoder(source, source_lengths)      # [1, B, H]
        decoder_input = target[:, 0]                       # <BOS>
        logits = []
        for t in range(1, target.size(1)):
            out, hidden = self.decoder(decoder_input, hidden)
            logits.append(out)
            if random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t]
            else:
                decoder_input = out.argmax(dim=-1)          # 模型自己的预测
        return torch.stack(logits, dim=1)                  # [B, T-1, V]


# ----------------------------------------------------------------------------
# 6. 训练与推理 (v0.1: Teacher Forcing 训练 + Greedy 解码)
# ----------------------------------------------------------------------------

def run_epoch(model, loader, optim, teacher_forcing_ratio, training):
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()
    total, n = 0.0, 0
    with ctx:
        for b in loader:
            src = b.source.to(device)
            lens = b.source_lengths.to(device)
            tgt = b.target.to(device)
            if training:
                optim.zero_grad(set_to_none=True)
            out = model(src, lens, tgt, teacher_forcing_ratio)     # [B, T-1, V]
            loss = torch.nn.functional.cross_entropy(
                out.reshape(-1, len(tgt_vocab)), tgt[:, 1:].reshape(-1),
                ignore_index=tgt_vocab.pad_id)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def greedy_translate(model, sentence: str, max_len: int = 20) -> str:
    model.eval()
    src_ids = src_vocab.encode(sentence.strip().split(), add_bos=False, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    lens = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
    hidden = model.encoder(src, lens)
    dec_in = torch.tensor([tgt_vocab.bos_id], dtype=torch.long, device=device)
    preds = []
    for _ in range(max_len):
        out, hidden = model.decoder(dec_in, hidden)
        tok = out.argmax(dim=-1)
        if tok.item() == tgt_vocab.eos_id:
            break
        preds.append(tok.item())
        dec_in = tok
    return " ".join(tgt_vocab.decode(preds))


if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    EMBED, HIDDEN = 64, 128
    encoder = Encoder(len(src_vocab), EMBED, HIDDEN, src_vocab.pad_id)
    decoder = Decoder(len(tgt_vocab), EMBED, HIDDEN, tgt_vocab.pad_id)
    model = Seq2Seq(encoder, decoder).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    train_loader = build_loader(4000, 64, shuffle=True)
    losses = []
    for epoch in range(1, 13):
        # 逐步降低 Teacher Forcing：从 1.0 到 0.25，模型逐步练习贪心解码
        tf = max(0.25, 1.0 - epoch * 0.08)
        loss = run_epoch(model, train_loader, optim, tf, True)
        losses.append(loss)
        print(f"epoch={epoch:02d} teacher_forcing={tf:.2f} train_loss={loss:.4f}")

    # v0.1 验收 1：训练 loss 明显下降
    assert losses[0] > losses[-1], f"loss did not decrease: {losses}"
    print(f"[PASS] loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # v0.1 验收 2：Greedy 解码能复现正确答案
    tests = ["three one nine", "zero five two seven", "eight eight four six one",
             "one two"]
    expected = {"three one nine": "三 一 九",
                "zero five two seven": "零 五 二 七",
                "eight eight four six one": "八 八 四 六 一",
                "one two": "一 二"}
    exact = 0
    for s in tests:
        p = greedy_translate(model, s)
        ok = p == expected[s]
        exact += int(ok)
        print(f"    input: {s!r:24} -> output: {p!r:18} {'[OK]' if ok else ''}")
    assert exact >= 4 and (exact == len(tests)), f"translation inaccurate ({exact}/{len(tests)})"
    print(f"[PASS] greedy decoding exact on {exact}/{len(tests)} test sentences")