# m03: 规范化批处理（v0.2）— Padding / Mask / pack_padded_sequence

[返回根目录](../README.md)

---

## The Problem

m02 的 MVP 虽然能用，训练却是**把每条样本的 PAD 也当成真实序列在算**：`nn.GRU` 会一路跑过 `<PAD>` 位置，既浪费算力，又会让 padding 的隐状态污染最后一步的 Context；同时 loss 里 `<PAD>` 位置也参与了梯度。`versions.md` 把 v0.2 定义为"规范化批处理"，要解决：

1. **Encoder 不该在 `<PAD>` 上继续算 RNN**——`pack_padded_sequence` 让 GRU 只算到每个样本的真实长度；
2. **Decoder 的 loss 不该把 `<PAD>` 位置计进去**——`ignore_index=pad_id`；
3. **一次要打很多样本，长度参差不齐**——需要 **Length Bucketing**，把长度相近的样本放进同一 batch，减少 padding 浪费。

## The Solution

三件事，对应三个断言：

```
pack_padded_sequence  →  Encoder 隐状态(取最后真 token) 与朴素 padded 一致  [A]
ignore_index=pad       →  把真 target 挖洞成 PAD 后 loss 与"手工只算非PAD"一致  [B]
length bucketing       →  同 batch 内源长度差 ≤ 4                          [C]
```

## How It Works

### 1. `pack_padded_sequence`：只跑真实长度

`nn.utils.rnn.pack_padded_sequence` 把 padded 序列压缩成 `PackedSequence`，GRU 内部按各样本真实步对齐，末尾 PAD 不再展开计算。编码器再用 `pad_packed_sequence` 展开回 `[B,S,H]`：

```python
packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
packed_out, _ = self.gru(packed)
out, _ = pad_packed_sequence(packed_out, batch_first=True)   # [B,S,H]
```

> **为什么 diff 不是 0**：packed 让 GRU 只跑到真实长度；naive 会继续跑过末尾的 PAD。两者取「最后一个真 token」隐状态时数值差在 `5e-5` 量级（fp32 归并顺序不同）——本校验设容差 `1e-3`，含义是「语义等价，而非位级相等」。

### 2. `ignore_index=pad`：PAD 位不计梯度

`F.cross_entropy(..., ignore_index=pad_id)` 会在求均值时**跳过**目标为 `pad_id` 的位置——既不算 loss、也不回传梯度。校验用「手工只对非 PAD 位置求 CE，结果必须与 `ignore_index` 版完全一致」来证明它确实等价。

### 3. Batch 内长度对齐：Bucketing

`bucket_sampler` 先把样本按源长度排序，再按 `batch_size` 顺序切块：

```python
order = sorted(range(len(dataset)), key=lambda i: len(dataset.examples[i][0]))
for start in range(0, len(order), batch_size):
    yield order[start:start + batch_size]
```

同一 batch 内长度相近 → PAD 列最少。

## Code Walkthrough

> 下面每行都对应 `code.py` 里的真实实现。

### Step 1: 带 Mask 的 `collate_bucket`

```python
src_mask[r, :length] = True       # 源 mask：真实 token
tgt_mask = (tgt_b != pad)         # 目标 mask：非 PAD 才监督
```

### Step 2: `PackedEncoder.forward`（v0.2 核心）

```python
emb = self.embed(src)
packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                              enforce_sorted=False)
packed_out, _ = self.gru(packed)
out, _ = pad_packed_sequence(packed_out, batch_first=True)
gather = torch.gather(out, 1, (lengths - 1).unsqueeze(1).expand(-1, -1, H))
return gather.squeeze(1).unsqueeze(0)            # [1, B, H]
```

### Step 3: `NaiveEncoder`（对照）— 什么都不包，直接跑 padded

只为了在 `[A]` 里证明「packed == naive(语义)」，不用于真正训练。

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **容差 1e-3 而非 1e-5** | packed / naive 是两条不同的归约路径，fp32 会产生 ~1e-5 量级舍入；断言目的是「语义一致」，不是「位级比特一致」。真正的等价比对放在逻辑层面。 |
| **Encoder 用 packed、Decoder loss 用 ignore_index** | 一个省算力、一个正确 loss——两件事分开做，各归各位。 |
| **Bucketing 采样器显式写** | 不引入深度工具，让"长度分桶"这个 v0.8 的关键手段在 v0.2 就可理解和复用。 |

## Going Deeper

- **pack 与可以不 pack**：`pad` 对 RNN 训练并不是"错误"，只是低效+轻微污染；若用 `bundle/` 或 `torch.compile`，pack 的收益会被掩盖——所以生产环境中常只在可观测关键路径上用 pack。
- **`enforce_sorted=False`**：允许加载器不按长度排序就 pack（省一次排序）。代价是 pack 内部仍要按长度重排，多一点点 GPU。教学里保持默认排序更直观。
- **下一步**：m04 的 Bahdanau Attention 需要每个 Encoder 位置的输出（不只是最后一个），因此 `PackedEncoder` 返回的 `out` 可以让 Attention 直接 `att = softmax(score) @ out`——v0.2 的 pack 已为 v0.3 的主架构升级铺好路。

## 模块定位

v0.2 补齐了"变长序列在多进程里到底怎么喂"的工程底座：`pack_padded_sequence` 省算力、`ignore_index` 校正 loss、Bucketing 对齐 batch。m04（v0.3）将在这一底座上，把"只取最后隐状态"升级为"每个位置都保留、供 Attention 加权"。