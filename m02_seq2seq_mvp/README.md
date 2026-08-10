[返回根目录](../README.md)

# m02: GRU Encoder-Decoder MVP — 英文数字序列→中文数字序列（v0.1）


---

## The Problem

有了 m01 的 GRU，我们想拼出第一个**能翻译的完整系统**：给定一段英文数字序列 `"three one nine"`，输出中文数字序列 `"三 一 九"`。注意输入和输出的长度**可以不同**，也不存在逐词对齐——这正是 Seq2Seq 与普通分类/序列标注的区别。

`versions.md` 把 v0.1 的目标定为"建立最小可运行闭环"，它需要同时具备：

- 独立的输入词表与输出词表（英文 vs 中文，token 集完全不同）
- 变长序列 + Padding
- Encoder + Decoder
- **Teacher Forcing**（训练时把正确答案喂回 Decoder）
- Cross Entropy + 梯度裁剪
- **Greedy Decoding**（推理时用自己的预测继续）

## The Solution

一个最小但完整的 **GRU Encoder-Decoder**，核心假设是**固定长度 Context**：整条输入被压缩成 Encoder 最终隐状态 `c = h_n`，Decoder 用这一个向量初始化，逐词吐输出。

```
Input  :"three one nine"
         ↓ Vocabulary.encode(+EOS)  →  [7, 5, 13, 2]
Embedding ↓
GRU Encoder ↓                          # outputs: [sequence, hidden]
  取最后一个非 PAD 步 h_n ↓
Context = h4                            # ← 整条输入压成一个向量
Decoder: <BOS> + h_n → "三" → "一" → "九" → <EOS>
```

## How It Works

### 1. Encoder：读入整条输入，吐出「最后一个隐状态」

`nn.GRU(embed_dim, hidden_dim, batch_first=True)` 处理整条序列，返回每步输出 `[B, S, H]`。因为带 Padding，不能直接取 `outputs[:, -1]`，要根据真实长度 `source_lengths` 取每个样本最后一个非 PAD 步：

```python
last_idx = (source_lengths - 1)
last_hidden = outputs[batch_idx, last_idx]  # [B, H]
hidden = last_hidden.unsqueeze(0)           # [num_layers=1, B, H]
```

### 2. Decoder：用一句话 + 上一步的 token/hidden，生成下一个 token

Decoder 的 GRU 是**单步**的：输入 `<BOS>` 和初始 `hidden=h_n`，得到 logits；把预测的 token 再喂回去，继续下一步。

```python
emb = embedding(input_token).unsqueeze(1)   # [B, 1, E]
out, next_hidden = gru(emb, hidden)          # [B, 1, H]
logits = out_projection(out.squeeze(1))      # [B, V]
```

### 3. Teacher Forcing：训练与推理的第一个分水岭

- **训练**：以概率 `teacher_forcing_ratio` 把**真实** target token 喂进 Decoder（指引正确方向）；
- **推理**：永远用**模型自己的**预测（`argmax`）喂回去。

```python
if random.random() < teacher_forcing_ratio:
    decoder_input = target[:, t]          # 正确答案
else:
    decoder_input = out.argmax(dim=-1)    # 模型自己的预测
```

这个「训练喂真值、推理喂预测」的差异，正是 `versions.md` v0.7 要解决的 **Exposure Bias**——现在先引入它，后面再修。

## Code Walkthrough

> 下面每行都对应 `code.py` 里的真实实现。

### Step 1: 词表 + Synthetic 数据（v0.1）

```python
src_vocab = Vocabulary(ENGLISH_NUMBERS)     # 独立输入词表
tgt_vocab = Vocabulary(CHINESE_NUMBERS)     # 独立输出词表
# Encoder 序列：[7, 5, 13, 2]（加 EOS）
# Decoder 序列：[<BOS>, 三, 一, 九, <EOS>]
# 损失对齐：模型第 t 步的输出 ⊂ 监督目标 target[:, t]（略过 <BOS>）
```

### Step 2: collate（变长 → padded batch）

```python
srcs, tgts = map(list, zip(*examples))
src_batch = pad_sequence(srcs, batch_first=True, padding_value=src_vocab.pad_id)
tgt_batch = pad_sequence(tgts, batch_first=True, padding_value=tgt_vocab.pad_id)
```

### Step 3: Seq2Seq.forward（v0.1）

```python
hidden = self.encoder(source, source_lengths)   # [1, B, H]
decoder_input = target[:, 0]                     # <BOS>
for t in range(1, target.size(1)):
    logits_t, hidden = self.decoder(decoder_input, hidden)
    logits.append(logits_t)                       # 收集逐 token logits
    if random.random() < teacher_forcing_ratio:
        decoder_input = target[:, t]
    else:
        decoder_input = logits_t.argmax(dim=-1)
return torch.stack(logits, dim=1)                # [B, T-1, V]
```

### Step 4: 训练（带梯度裁剪）+ `__main__` 自检

1. 训练 loss 从 **1.53 → 0.14** → `[PASS] loss decreased`
2. 对 4 条测试句 Greedy 解码 `exact 4/4` → `[PASS] greedy decoding exact`

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **两个独立词表** | 源/目标语言 token 完全不同（英文 vs 中文），必须分开建词表；这也是 v0.8 提到"共享 Embedding"时能省的前提。 |
| **Encoder 只取最后隐状态当 Context** | 这是 v0.1 的明确定义（`c = h_n`）。它简单可跑，但正是 v0.3 Attention 要打破的那个"固定向量瓶颈"。 |
| **Decoder 逐词（不乱用 Teacher Forcing 只在某些 token）** | 逐 step 循环让我们能看到每个生成位的中间状态；教学目的优于吞吐。 |
| **梯度裁剪 max_norm=1.0** | 循环网络 W_hh 连长，矩阵相乘特征值可能 >1 → 梯度爆炸。clip 是 m02 之后一直带的核心武器。 |

## Going Deeper

- **Teacher Forcing 的代价**：现在模型训练时见到的是真标签，推理时见到的是自己的错误——分布不一致就是 Exposure Bias，会让推理时错误被放大。v0.7 的 **Scheduled Sampling** 会逐步把真标签换成模型预测。
- **固定 Context 的局限**：输入越长，收尾信息越把开头信息「挤出去」。v0.3 Bahdanau Attention 让 Decoder 每步都重新回顾整条 Encoder 输出。
- **为什么 loss 能降到 0.14 但顺序还是会错**：这个 "一堆数字要做到顺序对" 的任务，在纯固定向量下对顺序敏感——decoding 一旦某位错，后续全错。这也解释了为什么 v1.1 非要 Transformer 这种「所有位置同时看全」的并行架构。

## 模块定位

m02 是通往 Seq2Seq 的**第一座可运行的桥**：它把 m01 的 GRU 组装成 Encoder + Decoder，并用玩具翻译任务证明「Teacher Forcing 训练的闭环 + 逐词 Greedy 解码」真的能翻译。m03 起，v0.2 开始处理"变长 batch 的规范化"。你已具备继续升级它（Attention / 双向 / Beam）的全部基础零件。