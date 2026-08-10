[返回根目录](../README.md)

# m07: Beam Search + 长度归一化 — 从局部最优到 Top-K 候选（v0.6）


---

## The Problem

m02 的 Decoder 推理用 **Greedy**：每步选 `argmax` 最高的 token。它的问题是**局部最优 ≠ 全局最优**：

```
第一步：
  A = 0.51，B = 0.49     ← 只看这一步，Greedy 选 A
后续：
  P(完整句 | A) 很低       ← 但选了 A，后面整句崩
  P(完整句 | B) 很高       ← B 才是对的路
```

`versions.md` 的 v0.6 目标：不只看单步 `argmax`，而是维护 **K 条候选序列**，逐步淘汰差的、保留好的，用**整个序列的累计概率**来选择。

## The Solution

**Beam Search**：每一步持有 K 个候选（"beam"），每个候选展开成 V 个下一 token，从 `K×V` 里再挑分数最高的 K 条。

```
step1: Top-K
  ↓
step2: 每个候选扩展整个词表 → K×V 个候选
  ↓
      重新保留 Top-K（按累计分数）
  ↓
...
```

序列分数 = `Σ log p(y_t | y_<t, x)`（log 概率累加，便于乘变成加）。

**长度归一化**（Google NMT）防止模型偏好短句——猜短句分数天然高：

```
S_norm(y) = S(y) / ((5 + |y|) / 6)^alpha
```

常见 `beam=4~5`、`alpha=0.6~1.0`。

## How It Works

### 关键设计一：每个 Beam 持有独立 hidden

不同前缀的 Decoder 隐状态不同——`[bos,3]` 的 hidden 和 `[bos,4]` 的 hidden 是两个不同张量。所以 beam 状态必须是 `(score, seq, hidden)` 三元组，扩展时互不干扰。

### 关键设计二：EOS 的处理

- 序列以 `<EOS>` 结束时**放进 finished**，不再扩展；
- 未结束的继续扩展；
- 最后在 finished（或未终止的）候选中按**长度归一分数**取最高。

### 关键设计三：长度归一化

`length_norm(len, α) = ((5+len)/6)^α`，随 `len` 单调增作为**分母**。
它会把"短但是次优"的候选在归一化后挤下去，从而公平比较不同长度的完整序列。

## Code Walkthrough

> 下面每行都对应 `code.py` 里的真实实现。

### Step 1: 长度归一化

```python
def length_norm(seq_len, alpha=0.6):
    return ((5 + seq_len) / 6) ** alpha
```

### Step 2: Beam Search 主循环

```python
beams = [(0.0, [bos], init_hidden)]      # (score, seq, hidden)
for _ in range(max_len):
    expanded = []
    for score, seq, hidden in beams:
        logits, next_hidden = step_fn(seq[-1], hidden)
        probs = F.log_softmax(logits, -1).squeeze(0)
        for nxt in range(vocab):
            expanded.append((score + probs[nxt].item(), seq + [nxt], next_hidden))
    expanded.sort(key=norm_key, reverse=True)   # 按长度归一分数
    beams = []
    for score, seq, hidden in expanded[:beam]:
        if seq[-1] == eos: finished.append((score, seq))
        else: beams.append((score, seq, hidden))
```

### Step 3: `__main__` 自检（用确定性假 step_fn，不依赖网络收敛）

构造一个已知全局最优 `golden=[3,4,5,eos]` 的假解码器，每次把 `golden[t]` 的 logit 拉到 7.0：

1. **Beam 找到 golden**（顺序正确、以 EOS 收尾）→ `[PASS] beam search returns golden sequence`
2. **长度归一单调**：`length_norm(3)<length_norm(8)` → `[PASS] length_norm monotonic`
3. **长度归一压缩短句偏差**：`(len2,-2.0)` 与 `(len4,-2.6)` 归一化后分数差被拉近 → `[PASS] reduces short-seq bias`

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **每个 beam 独立 hidden** | 不同前缀的 decoder 状态不同；共享 hidden 会串味，必须 `(score, seq, hidden)` 一起存。versions.md 特意强调"每个 Beam 都要维护独立 Hidden"。 |
| **log 分数累加而非概率相乘** | 对数把连乘变连加，数值更稳（避免大量极小概率下溢），且单调序不变。 |
| **只留 Top-K 而非全部** | 复杂度从 `V^T` 降到 `O(K·V·T)`；K 越大越接近全局最优，但生成越"安全平庸"。 |
| **长度归一在排序与最终选择都生效** | 全程用同一长度归一分数，避免在扩展/最终选择两处标准不一致。 |

## Going Deeper

- **Beam 越大不一定越好**：大的 beam 倾向于流畅安全但平庸的文本，小的 beam 更容易冒进；挑时代对 `alpha` 调优。
- **Gotcha：正确处理 EOS**：已 EOS 的假设不能再扩展，否则分数可能被继续累加导致不同长度无法公平比较。
- **m07 与 Exposure Bias 的关系**：Beam Search 是在**推理期**改进；训练期喂真值、推理期喂自己的产物之间的不一致（Exposure Bias）交给 v0.7 的 Scheduled Sampling 解决。两者顺序符合 `versions.md`：先做解码搜索（v0.6），再做训练采样（v0.7）。

## 模块定位

m07 把解码从"单步 argmax"升级到"K 候选 + 长度归一"，并给你一个**不依赖网络收敛、可确定性验证**的 beam 实现。m08（v0.7）回头补训练侧的 Exposure Bias 与过度自信。