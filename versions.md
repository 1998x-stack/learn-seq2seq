# 一、先澄清发展关系

严格来说，下面并不是四个完全同级、线性替代的模型：

```text
RNN
 ├─ LSTM：一种改进的 RNN Cell
 └─ GRU：另一种改进的 RNN Cell

RNN / LSTM / GRU
        ↓ 作为编码器和解码器组件
     Seq2Seq
        ↓
 Seq2Seq + Attention
        ↓
    Transformer
```

因此，更准确的发展路线是：

[
\text{Vanilla RNN}
\rightarrow
\begin{cases}
\text{LSTM}\
\text{GRU}
\end{cases}
\rightarrow
\text{Encoder-Decoder Seq2Seq}
\rightarrow
\text{Attention Seq2Seq}
\rightarrow
\text{Transformer}
]

---

# 二、RNN → LSTM → GRU → Seq2Seq 发展表

| 阶段                | 核心结构                          | 主要解决的问题          | 关键机制                 | 优点            | 主要局限                    |
| ----------------- | ----------------------------- | ---------------- | -------------------- | ------------- | ----------------------- |
| Vanilla RNN       | 单隐状态递归                        | 对序列顺序和上下文建模      | (h_t=f(x_t,h_{t-1})) | 结构简单、参数少      | 梯度消失/爆炸，长期依赖差           |
| LSTM              | Cell State + 三个门              | 长距离依赖和梯度消失       | 输入门、遗忘门、输出门          | 长期记忆能力强、稳定    | 参数量大，计算较慢               |
| GRU               | 单状态 + 两个门                     | 简化 LSTM，保持长期记忆   | 更新门、重置门              | 参数更少、速度更快     | 某些复杂长序列任务略弱于 LSTM       |
| Seq2Seq           | Encoder + Decoder             | 输入输出长度不同的序列映射    | 编码器压缩输入，解码器逐步生成      | 支持翻译、摘要、对话    | 固定长度 Context Vector 是瓶颈 |
| Attention Seq2Seq | Encoder + Attention + Decoder | 长输入的信息压缩瓶颈       | 每个解码位置动态访问编码状态       | 长序列效果显著提升，可对齐 | 仍然顺序计算，训练较慢             |
| Transformer       | Self-Attention                | RNN 无法并行和长距离路径过长 | 自注意力、位置编码            | 高并行、扩展能力强     | Attention 的长度复杂度较高      |

---

# 三、各阶段核心原理

## 3.1 Vanilla RNN

RNN 在时间步 (t) 更新隐藏状态：

[
h_t=\tanh(W_xx_t+W_hh_{t-1}+b)
]

输出为：

[
y_t=W_oh_t+b_o
]

展开后：

```text
x1 → RNN → h1
           ↓
x2 → RNN → h2
           ↓
x3 → RNN → h3
```

问题在于反向传播需要连续乘很多次矩阵：

[
\frac{\partial h_T}{\partial h_t}
=================================

\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}
]

当特征值小于 1 时，梯度快速趋近于零；大于 1 时，梯度可能爆炸。

---

## 3.2 LSTM

LSTM 引入独立的 Cell State：

[
c_t=f_t\odot c_{t-1}+i_t\odot \tilde c_t
]

其中：

### 遗忘门

[
f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)
]

决定保留多少旧记忆。

### 输入门

[
i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)
]

决定写入多少新信息。

### 候选记忆

[
\tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c)
]

### 输出门

[
o_t=\sigma(W_o[x_t,h_{t-1}]+b_o)
]

### 隐状态

[
h_t=o_t\odot\tanh(c_t)
]

关键变化是 Cell State 上存在相对直接的加法路径，使梯度更容易传播。

---

## 3.3 GRU

GRU 将 LSTM 的 Cell State 和 Hidden State 合并，只有两个主要门。

### 更新门

[
z_t=\sigma(W_zx_t+U_zh_{t-1})
]

控制保留多少历史状态。

### 重置门

[
r_t=\sigma(W_rx_t+U_rh_{t-1})
]

控制计算候选状态时使用多少历史信息。

### 候选状态

[
\tilde h_t=
\tanh(W_hx_t+U_h(r_t\odot h_{t-1}))
]

### 最终状态

[
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t
]

粗略参数量对比：

| Cell | 主要变换数量 | 相对参数量 |
| ---- | -----: | ----: |
| RNN  |      1 |    1× |
| GRU  |      3 |    3× |
| LSTM |      4 |    4× |

GRU 通常是实现 Seq2Seq MVP 的良好选择：比 RNN 稳定，比 LSTM 简洁。

---

# 四、Seq2Seq 的核心架构

Seq2Seq 将输入序列映射为另一个长度可能不同的输出序列：

[
(x_1,x_2,\ldots,x_n)
\rightarrow
(y_1,y_2,\ldots,y_m)
]

## 4.1 Encoder

编码输入序列：

[
h_t=\operatorname{GRU}(x_t,h_{t-1})
]

最终得到上下文向量：

[
c=h_n
]

## 4.2 Decoder

Decoder 使用 Encoder 最后的隐藏状态初始化：

[
s_0=c
]

逐个生成输出：

[
p(y_t\mid y_{<t},x)
===================

\operatorname{softmax}(W_os_t)
]

整体概率分解：

[
p(y\mid x)
==========

\prod_{t=1}^{m}
p(y_t\mid y_{<t},x)
]

## 4.3 特殊 Token

```text
<BOS>：输出序列开始
<EOS>：输出序列结束
<PAD>：批处理补齐
<UNK>：未知词
```

训练时：

```text
Decoder 输入：<BOS> 一 二 三
监督目标：      一 二 三 <EOS>
```

---

# 五、Seq2Seq MVP：GRU Encoder-Decoder

下面实现一个最小的“英文数字序列 → 中文数字序列”任务。

示例：

```text
输入：three one nine
输出：三 一 九
```

它具备完整 Seq2Seq 的核心组件：

* 独立输入词表和输出词表；
* 变长序列；
* Padding；
* Encoder；
* Decoder；
* Teacher Forcing；
* Cross Entropy；
* Greedy Decoding；
* 梯度裁剪。

```python
# ============================================================
# Seq2Seq MVP: GRU Encoder-Decoder
# Task: English number sequence -> Chinese number sequence
# ============================================================

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ------------------------------------------------------------
# 2. Vocabulary
# ------------------------------------------------------------

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

ENGLISH_NUMBERS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

CHINESE_NUMBERS = [
    "零",
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
]


class Vocabulary:
    def __init__(self, tokens: List[str]):
        all_tokens = SPECIAL_TOKENS + tokens

        self.token_to_id = {
            token: index
            for index, token in enumerate(all_tokens)
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

    def encode(
        self,
        tokens: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]

        if add_bos:
            ids = [self.bos_id] + ids

        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(
        self,
        ids: List[int],
        remove_special_tokens: bool = True,
    ) -> List[str]:
        tokens = []

        for token_id in ids:
            token = self.id_to_token.get(token_id, "<UNK>")

            if token == "<EOS>":
                break

            if remove_special_tokens and token in SPECIAL_TOKENS:
                continue

            tokens.append(token)

        return tokens

    def __len__(self) -> int:
        return len(self.token_to_id)


src_vocab = Vocabulary(ENGLISH_NUMBERS)
tgt_vocab = Vocabulary(CHINESE_NUMBERS)

number_mapping = dict(zip(ENGLISH_NUMBERS, CHINESE_NUMBERS))


# ------------------------------------------------------------
# 3. Synthetic sequence-to-sequence dataset
# ------------------------------------------------------------

def generate_example(
    min_length: int = 1,
    max_length: int = 7,
) -> Tuple[List[str], List[str]]:
    length = random.randint(min_length, max_length)

    source_tokens = [
        random.choice(ENGLISH_NUMBERS)
        for _ in range(length)
    ]

    target_tokens = [
        number_mapping[token]
        for token in source_tokens
    ]

    return source_tokens, target_tokens


class NumberTranslationDataset(Dataset):
    def __init__(self, size: int):
        self.examples = [
            generate_example()
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        source_tokens, target_tokens = self.examples[index]

        # Encoder 不需要 BOS，但加入 EOS。
        source_ids = src_vocab.encode(
            source_tokens,
            add_bos=False,
            add_eos=True,
        )

        # Decoder 需要 BOS 和 EOS。
        target_ids = tgt_vocab.encode(
            target_tokens,
            add_bos=True,
            add_eos=True,
        )

        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


@dataclass
class Batch:
    source: torch.Tensor
    source_lengths: torch.Tensor
    target: torch.Tensor


def collate_batch(
    examples: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Batch:
    source_sequences, target_sequences = zip(*examples)

    source_lengths = torch.tensor(
        [len(sequence) for sequence in source_sequences],
        dtype=torch.long,
    )

    source_batch = pad_sequence(
        source_sequences,
        batch_first=True,
        padding_value=src_vocab.pad_id,
    )

    target_batch = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=tgt_vocab.pad_id,
    )

    return Batch(
        source=source_batch,
        source_lengths=source_lengths,
        target=target_batch,
    )


train_dataset = NumberTranslationDataset(size=5000)
validation_dataset = NumberTranslationDataset(size=500)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_batch,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_batch,
)


# ------------------------------------------------------------
# 4. Encoder
# ------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_id: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        source:
            [batch_size, source_length]

        return:
            hidden [1, batch_size, hidden_dim]
        """

        embedded = self.embedding(source)

        # outputs:
        # [batch_size, source_length, hidden_dim]
        outputs, _ = self.gru(embedded)

        # 找到每个样本最后一个非 PAD 时间步。
        last_indices = source_lengths.to(source.device) - 1

        batch_indices = torch.arange(
            source.size(0),
            device=source.device,
        )

        last_hidden = outputs[
            batch_indices,
            last_indices,
        ]

        # GRU hidden 格式：
        # [num_layers, batch_size, hidden_dim]
        hidden = last_hidden.unsqueeze(0)

        return hidden


# ------------------------------------------------------------
# 5. Decoder
# ------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_id: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_token:
            [batch_size]

        hidden:
            [1, batch_size, hidden_dim]

        return:
            logits [batch_size, vocab_size]
            next_hidden [1, batch_size, hidden_dim]
        """

        embedded = self.embedding(input_token)

        # 添加长度为 1 的时间维。
        embedded = embedded.unsqueeze(1)

        output, next_hidden = self.gru(
            embedded,
            hidden,
        )

        # output:
        # [batch_size, 1, hidden_dim]
        output = output.squeeze(1)

        logits = self.output_projection(output)

        return logits, next_hidden


# ------------------------------------------------------------
# 6. Seq2Seq
# ------------------------------------------------------------

class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        target:
            [batch_size, target_length]

        returns:
            logits [batch_size, target_length - 1, target_vocab_size]
        """

        hidden = self.encoder(
            source,
            source_lengths,
        )

        # 第一个输入是 <BOS>。
        decoder_input = target[:, 0]

        all_logits = []

        # t=1 时预测 target[:, 1]。
        for time_step in range(1, target.size(1)):
            logits, hidden = self.decoder(
                decoder_input,
                hidden,
            )

            all_logits.append(logits)

            predicted_token = logits.argmax(dim=-1)

            use_teacher_forcing = (
                random.random() < teacher_forcing_ratio
            )

            if use_teacher_forcing:
                decoder_input = target[:, time_step]
            else:
                decoder_input = predicted_token

        return torch.stack(all_logits, dim=1)


# ------------------------------------------------------------
# 7. Model initialization
# ------------------------------------------------------------

EMBEDDING_DIM = 64
HIDDEN_DIM = 128

encoder = Encoder(
    vocab_size=len(src_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    pad_id=src_vocab.pad_id,
)

decoder = Decoder(
    vocab_size=len(tgt_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    pad_id=tgt_vocab.pad_id,
)

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-3,
)

criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_vocab.pad_id,
)

print(model)


# ------------------------------------------------------------
# 8. Train and evaluate
# ------------------------------------------------------------

def run_epoch(
    data_loader: DataLoader,
    training: bool,
    teacher_forcing_ratio: float,
) -> float:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in data_loader:
            source = batch.source.to(device)
            source_lengths = batch.source_lengths.to(device)
            target = batch.target.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(
                source=source,
                source_lengths=source_lengths,
                target=target,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            # logits:
            # [batch_size, target_length - 1, vocab_size]
            #
            # target[:, 1:]:
            # 移除 <BOS>，与预测位置对齐。
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target[:, 1:].reshape(-1),
            )

            if training:
                loss.backward()

                # 防止循环网络梯度爆炸。
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


EPOCHS = 8

for epoch in range(1, EPOCHS + 1):
    # 可以逐步降低 Teacher Forcing。
    teacher_forcing_ratio = max(
        0.2,
        1.0 - epoch * 0.1,
    )

    train_loss = run_epoch(
        train_loader,
        training=True,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )

    validation_loss = run_epoch(
        validation_loader,
        training=False,
        teacher_forcing_ratio=0.0,
    )

    print(
        f"epoch={epoch:02d} "
        f"teacher_forcing={teacher_forcing_ratio:.2f} "
        f"train_loss={train_loss:.4f} "
        f"validation_loss={validation_loss:.4f}"
    )


# ------------------------------------------------------------
# 9. Greedy decoding
# ------------------------------------------------------------

@torch.no_grad()
def translate(
    sentence: str,
    max_output_length: int = 20,
) -> str:
    model.eval()

    source_tokens = sentence.strip().split()

    source_ids = src_vocab.encode(
        source_tokens,
        add_bos=False,
        add_eos=True,
    )

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device,
    )

    source_lengths = torch.tensor(
        [len(source_ids)],
        dtype=torch.long,
        device=device,
    )

    hidden = model.encoder(
        source,
        source_lengths,
    )

    decoder_input = torch.tensor(
        [tgt_vocab.bos_id],
        dtype=torch.long,
        device=device,
    )

    generated_ids = []

    for _ in range(max_output_length):
        logits, hidden = model.decoder(
            decoder_input,
            hidden,
        )

        next_token = logits.argmax(dim=-1)
        next_token_id = next_token.item()

        if next_token_id == tgt_vocab.eos_id:
            break

        generated_ids.append(next_token_id)
        decoder_input = next_token

    output_tokens = tgt_vocab.decode(generated_ids)

    return " ".join(output_tokens)


# ------------------------------------------------------------
# 10. Test
# ------------------------------------------------------------

test_sentences = [
    "three one nine",
    "zero five two seven",
    "eight eight four six one",
]

for sentence in test_sentences:
    prediction = translate(sentence)

    print(f"input : {sentence}")
    print(f"output: {prediction}")
    print("-" * 40)
```

---

# 六、MVP 中的数据流

假设输入：

```text
three one nine
```

编码后：

```text
three → 7
one   → 5
nine  → 13
<EOS> → 2
```

Encoder：

```text
[7, 5, 13, 2]
    ↓ Embedding
[e1, e2, e3, e4]
    ↓ GRU
[h1, h2, h3, h4]
    ↓
Context = h4
```

Decoder：

```text
<BOS> + h4
    ↓
预测“三”
    ↓
“三” + hidden
    ↓
预测“一”
    ↓
“一” + hidden
    ↓
预测“九”
    ↓
预测 <EOS>
```

---

# 七、Seq2Seq 各版本迭代路线

## 总览

| 版本   | 核心升级                               | 主要解决的问题                |
| ---- | ---------------------------------- | ---------------------- |
| v0.1 | Vanilla GRU Encoder-Decoder        | 建立最小可运行闭环              |
| v0.2 | Batch、Padding、Mask、长度处理            | 支持变长序列和批训练             |
| v0.3 | Bahdanau Attention                 | 固定 Context Vector 瓶颈   |
| v0.4 | 双向 Encoder                         | 让编码位置同时看到左右上下文         |
| v0.5 | 多层 GRU/LSTM、Dropout、Residual       | 提升模型容量和泛化能力            |
| v0.6 | Beam Search、长度归一化                  | 改善推理解码质量               |
| v0.7 | Scheduled Sampling、Label Smoothing | 缓解 Exposure Bias 和过度自信 |
| v0.8 | Subword、Bucketing、Shared Embedding | 面向真实 NLP 数据            |
| v0.9 | Copy / Pointer、Coverage            | OOV、事实复制、重复生成          |
| v1.0 | AMP、DDP、Checkpoint、评测体系            | 面向生产训练                 |
| v1.1 | Transformer Encoder-Decoder        | 移除循环依赖，实现并行训练          |

---

# 八、v0.1：最小 Encoder-Decoder

结构：

```text
Input
  ↓
Embedding
  ↓
GRU Encoder
  ↓
最后隐藏状态 h_n
  ↓
GRU Decoder
  ↓
Linear + Softmax
```

核心假设：

[
c=h_n
]

整个输入都被压缩成一个固定长度向量。

适合：

* 理解 Seq2Seq；
* 短序列；
* 玩具翻译；
* 日期、数字和简单格式转换。

主要问题：

```text
输入长度增加
    ↓
所有信息仍被压入一个向量
    ↓
早期信息逐渐丢失
```

---

# 九、v0.2：规范化批处理

需要增加：

1. Padding；
2. `ignore_index=PAD_ID`；
3. 真实长度；
4. Decoder 输出 Mask；
5. Bucket by length；
6. 可选 `pack_padded_sequence`。

损失函数：

[
\mathcal L
==========

-\sum_{t=1}^{m}
M_t\log p(y_t\mid y_{<t},x)
]

其中：

[
M_t=
\begin{cases}
1,& y_t\neq PAD\
0,& y_t=PAD
\end{cases}
]

MVP 已经实现了基础 Padding 和真实长度处理。

正式版本可将 Encoder 改为：

```python
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

packed = pack_padded_sequence(
    embedded,
    lengths.cpu(),
    batch_first=True,
    enforce_sorted=False,
)

packed_outputs, hidden = self.gru(packed)

outputs, _ = pad_packed_sequence(
    packed_outputs,
    batch_first=True,
)
```

这样 RNN 不会在 `<PAD>` 上继续计算。

---

# 十、v0.3：Bahdanau Attention

这是 Seq2Seq 最重要的一次升级。

## 10.1 固定 Context 的问题

原始 Seq2Seq：

[
c=h_n
]

Decoder 每个时间步都只使用相同的 (c)。

Attention 改为：

[
c_t=\sum_{i=1}^{n}\alpha_{t,i}h_i
]

每个 Decoder 时间步都有自己的 Context。

## 10.2 Bahdanau Attention

对齐分数：

[
e_{t,i}
=======

v^\top
\tanh(
W_hh_i+W_ss_{t-1}
)
]

归一化：

[
\alpha_{t,i}
============

\frac{\exp(e_{t,i})}
{\sum_j\exp(e_{t,j})}
]

上下文：

[
c_t=
\sum_i\alpha_{t,i}h_i
]

Decoder 输入：

[
s_t=
\operatorname{GRU}
([E(y_{t-1});c_t],s_{t-1})
]

结构：

```text
Encoder states:
h1  h2  h3  h4  h5
 \   |   |   |  /
  Attention at decoder step t
             ↓
            c_t
             ↓
previous token → Decoder → output token
```

核心 Attention 模块：

```python
class BahdanauAttention(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int,
    ):
        super().__init__()

        self.encoder_projection = nn.Linear(
            encoder_dim,
            attention_dim,
            bias=False,
        )

        self.decoder_projection = nn.Linear(
            decoder_dim,
            attention_dim,
            bias=False,
        )

        self.score_projection = nn.Linear(
            attention_dim,
            1,
            bias=False,
        )

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ):
        """
        decoder_hidden:
            [batch, decoder_dim]

        encoder_outputs:
            [batch, source_length, encoder_dim]

        source_mask:
            [batch, source_length]
        """

        encoder_features = self.encoder_projection(
            encoder_outputs
        )

        decoder_features = self.decoder_projection(
            decoder_hidden
        ).unsqueeze(1)

        energy = torch.tanh(
            encoder_features + decoder_features
        )

        scores = self.score_projection(
            energy
        ).squeeze(-1)

        scores = scores.masked_fill(
            ~source_mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)

        return context, attention_weights
```

升级后 Encoder 必须返回所有位置：

```python
encoder_outputs, hidden = encoder(...)
```

而不只是最后一个隐藏状态。

---

# 十一、v0.4：双向 Encoder

单向 Encoder：

[
h_i=f(x_i,h_{i-1})
]

位置 (i) 只能看到左侧内容。

双向 Encoder：

[
\overrightarrow h_i
===================

f(x_i,\overrightarrow h_{i-1})
]

[
\overleftarrow h_i
==================

f(x_i,\overleftarrow h_{i+1})
]

组合：

[
h_i=
[
\overrightarrow h_i;
\overleftarrow h_i
]
]

PyTorch：

```python
self.gru = nn.GRU(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    batch_first=True,
    bidirectional=True,
)
```

输出维度变成：

```text
[batch, source_length, 2 × hidden_dim]
```

需要 Bridge 将双向 Encoder 状态映射到 Decoder：

```python
self.bridge = nn.Linear(
    hidden_dim * 2,
    decoder_hidden_dim,
)

decoder_init = torch.tanh(
    self.bridge(
        torch.cat(
            [forward_hidden, backward_hidden],
            dim=-1,
        )
    )
)
```

优点：

* 每个词同时编码左右上下文；
* 翻译和摘要通常更好；
* Attention 的 Key/Value 信息更完整。

限制：

* 不适合严格流式编码；
* 计算量约增加一倍。

---

# 十二、v0.5：深层 Seq2Seq

升级内容：

```text
Single-layer GRU
        ↓
Multi-layer GRU/LSTM
        +
Dropout
        +
Residual Connection
        +
Layer Normalization
```

多层 RNN：

```python
self.gru = nn.GRU(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    num_layers=3,
    dropout=0.2,
    batch_first=True,
)
```

注意：PyTorch RNN 的 `dropout` 只作用在层与层之间；当 `num_layers=1` 时不会生效。

深层模型的问题：

* 梯度传播更困难；
* Encoder 和 Decoder 层数可能不同；
* Hidden State Bridge 更复杂；
* 容易过拟合小数据。

因此通常配合：

```text
Gradient Clipping
Dropout
Residual Connection
Layer Normalization
Learning Rate Schedule
```

---

# 十三、v0.6：从 Greedy Search 到 Beam Search

## 13.1 Greedy Search

每一步选择概率最大的 Token：

[
y_t=\arg\max_y p(y\mid y_{<t},x)
]

问题是局部最优不等于全局最优。

例如：

```text
第一步：
A = 0.51
B = 0.49

后续：
P(sequence | A) 很低
P(sequence | B) 很高
```

Greedy 会永久选择 A。

## 13.2 Beam Search

维护 (K) 个候选序列：

```text
step 1: 保留 Top-K
step 2: 每个候选扩展整个词表
step 3: 从 K×V 个候选中重新保留 Top-K
...
```

序列分数：

[
S(y)=
\sum_t\log p(y_t\mid y_{<t},x)
]

需要长度归一化，否则模型偏好短输出：

[
S_{\text{norm}}(y)
==================

\frac{S(y)}
{\left(\frac{5+|y|}{6}\right)^\alpha}
]

常见设置：

```text
beam_size = 4 或 5
length_penalty = 0.6～1.0
```

Gotcha：

* Beam 越大不一定越好；
* 可能生成安全但平庸的文本；
* 必须正确处理已完成序列；
* 每个 Beam 都要维护独立 Hidden State。

---

# 十四、v0.7：解决 Exposure Bias

训练时使用真实历史 Token：

[
p(y_t\mid y_{<t}^{gold},x)
]

推理时使用模型自己的历史预测：

[
p(y_t\mid \hat y_{<t},x)
]

这造成训练和推理分布不一致，即 Exposure Bias。

## 14.1 Teacher Forcing

```text
训练：
正确答案 token → Decoder

推理：
模型预测 token → Decoder
```

## 14.2 Scheduled Sampling

逐步降低 Teacher Forcing 比例：

```python
teacher_forcing_ratio = max(
    minimum_ratio,
    initial_ratio * decay_rate ** epoch,
)
```

例如：

```text
epoch 1: 1.00
epoch 2: 0.90
epoch 3: 0.80
...
epoch 8: 0.30
```

但 Scheduled Sampling 并非万能方法，可能引入不一致训练目标。实践中应通过验证集比较，而不是默认一定提升。

## 14.3 Label Smoothing

普通目标：

```text
正确类别概率 = 1
其他类别概率 = 0
```

Label Smoothing：

[
q(y)=
\begin{cases}
1-\epsilon,&y=y^*\
\epsilon/(V-1),&y\neq y^*
\end{cases}
]

PyTorch：

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=pad_id,
    label_smoothing=0.1,
)
```

作用：

* 降低过度自信；
* 改善泛化；
* 有时提高 Beam Search 结果。

---

# 十五、v0.8：真实 NLP 数据版本

玩具数据升级到真实翻译，需要增加：

## 15.1 Subword Tokenizer

不再使用简单空格分词，而使用：

* BPE；
* WordPiece；
* Unigram；
* SentencePiece。

原因：

```text
词级 Vocabulary：
词表巨大
OOV 严重
低频词学习困难

字符级 Vocabulary：
序列过长
语义单元过细

Subword：
在词表大小与序列长度之间折中
```

## 15.2 Length Bucketing

把长度接近的样本放进同一 Batch：

```text
Batch A：长度 5～10
Batch B：长度 11～20
Batch C：长度 21～40
```

减少 Padding 浪费。

## 15.3 Shared Embedding

当源语言和目标语言词表相近时，可以共享：

```python
source_embedding.weight
target_embedding.weight
output_projection.weight
```

称为 Weight Tying。

## 15.4 真实评测

不能只看训练 Loss。

机器翻译常见指标：

| 指标          | 作用           |
| ----------- | ------------ |
| BLEU        | n-gram 精确匹配  |
| chrF        | 字符级匹配，多语言更稳健 |
| TER         | 编辑距离         |
| COMET       | 基于模型的语义质量评价  |
| Exact Match | 结构化任务或短答案    |

---

# 十六、v0.9：Copy 和 Coverage

## 16.1 Pointer-Generator

摘要和问答中经常需要复制输入中的：

* 人名；
* 地名；
* 数字；
* 专有名词；
* 罕见实体。

普通 Decoder 只能从固定词表生成。

Pointer-Generator 混合两种概率：

[
P(w)
====

p_{\text{gen}}P_{\text{vocab}}(w)
+
(1-p_{\text{gen}})P_{\text{copy}}(w)
]

其中：

* (P_{\text{vocab}})：从词表生成；
* (P_{\text{copy}})：通过 Attention 从输入复制。

## 16.2 Coverage

Attention Seq2Seq 容易重复生成：

```text
the model is useful
the model is useful
the model is useful
```

定义历史 Attention 累积：

[
coverage_{t,i}
==============

\sum_{t'<t}\alpha_{t',i}
]

在当前 Attention 中加入 Coverage：

[
e_{t,i}
=======

v^\top\tanh(
W_hh_i+
W_ss_{t-1}+
W_ccoverage_{t,i}
)
]

Coverage Loss：

[
\mathcal L_{\text{cov}}
=======================

\sum_i
\min(
\alpha_{t,i},
coverage_{t,i}
)
]

它鼓励模型不要反复关注已经生成过的输入区域。

---

# 十七、v1.0：生产训练版本

生产级 Seq2Seq 需要补齐：

```text
数据版本管理
Tokenizer 固定版本
训练/验证/测试切分
断点续训
混合精度 AMP
多 GPU DDP
梯度累积
学习率调度
Early Stopping
Beam Search
评测指标
实验日志
推理服务
```

训练状态至少保存：

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "best_validation_loss": best_validation_loss,
    "source_vocab": src_vocab.token_to_id,
    "target_vocab": tgt_vocab.token_to_id,
}
```

AMP：

```python
scaler = torch.amp.GradScaler("cuda")

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    logits = model(...)
    loss = criterion(...)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

scaler.step(optimizer)
scaler.update()
```

---

# 十八、v1.1：从 Attention Seq2Seq 到 Transformer

Attention Seq2Seq 已经让 Decoder 动态访问所有 Encoder 状态，但 Encoder 和 Decoder 内部仍然顺序计算：

[
h_t=f(h_{t-1},x_t)
]

这导致：

* 无法在时间维并行；
* 长距离信息需要经过多个递归步骤；
* GPU 利用率不高。

Transformer 将 RNN 替换为 Self-Attention：

[
\operatorname{Attention}(Q,K,V)
===============================

\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
]

发展关系：

```text
Seq2Seq：
固定向量承载全部输入信息

Attention Seq2Seq：
Decoder 动态访问 Encoder 全部状态

Transformer：
Encoder 和 Decoder 内部也全部改用 Attention
```

因此 Transformer 不是突然出现的全新思想，而是将 Seq2Seq Attention 推向了彻底的非递归化。

---

# 十九、建议的学习实现顺序

| 顺序 | 实现内容               | 学习目标                 |
| -: | ------------------ | -------------------- |
|  1 | 单层 RNN Cell 手写     | 理解时间递归和 BPTT         |
|  2 | 手写 LSTM Cell       | 理解门控和 Cell State     |
|  3 | 手写 GRU Cell        | 理解简化门控               |
|  4 | GRU Seq2Seq MVP    | 建立完整 Encoder-Decoder |
|  5 | Padding、Mask、Pack  | 掌握变长批处理              |
|  6 | Bahdanau Attention | 理解动态 Context         |
|  7 | 双向 Encoder         | 理解上下文增强              |
|  8 | Beam Search        | 理解序列级搜索              |
|  9 | Subword + 真实数据     | 转向实际 NLP             |
| 10 | Transformer        | 理解 Seq2Seq 架构的现代化    |

最关键的主线可以压缩为：

```text
RNN：
让神经网络拥有序列状态
    ↓
LSTM / GRU：
让状态能够保存更长期的信息
    ↓
Seq2Seq：
用两个循环网络完成变长序列映射
    ↓
Attention：
不再强迫所有输入信息进入单个向量
    ↓
Transformer：
不再使用循环，直接让所有位置相互交互
```

