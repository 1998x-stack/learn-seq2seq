[返回根目录](../README.md)

# m05: 双向 Encoder + Bridge — 让每个位置同时看到左右上下文（v0.4）


---

## The Problem

m04 的 Encoder 是**单向**的：`h_i = f(x_i, h_{i-1})`，位置 `i` 只能看到它**左侧**的内容。可很多翻译/摘要场景里，一个词的含义要**左右都看**才知道——`bank` 在 river bank 和 money bank 里意思完全不同。

`versions.md` 的 v0.4 目标：让 Encoder 的每个位置同时聚合**前向**和**后向**的上下文。

## The Solution

把 Encoder 改成**双向 GRU**：

```
→h_i = f(x_i, →h_{i-1})      前向：从左到右
←h_i = f(x_i, ←h_{i+1})      后向：从右到左
h_i  = [ →h_i ; ←h_i ]        拼接 → 每个位置同时"看"左右
```

因为 Encoder 输出维度翻倍到 `2H`，多出一个 **Bridge**：把双向终态 `[→h_n; ←h_1]` 通过一个 `Linear(2H→H) + tanh` 映射成 Decoder 初始隐状态。

```
Decoder 初始状态 = tanh( Bridge( [ 前向终态 ; 后向终态 ] ) )
```

## How It Works

`nn.GRU(bidirectional=True)` 一次算两个方向：输出 `[B, S, 2H]`（前向维度在前、后向在后），返回的 `hidden` 是 `[2, B, H]`——第 0 层是前向终态，第 1 层是后向终态：

```python
outputs, hidden = self.gru(emb)     # [B,S,2H]、[2,B,H]
fwd, bwd = hidden[0], hidden[1]      # 各 [B,H]
last = torch.cat([fwd, bwd], -1)     # [B,2H]  作为 Bridge 输入
```

Bridge 把维度压回 Decoder 的 H：

```python
class Bridge(nn.Module):
    def forward(self, enc_last):
        return torch.tanh(self.proj(enc_last)).unsqueeze(0)   # [1,B,H]
```

## Code Walkthrough

> 下面每行都对应 `code.py` 里的真实实现。

### Step 1: `BidirectionalEncoder`（v0.4 核心）

```python
self.gru = nn.GRU(embed_dim, hidden, bidirectional=True, batch_first=True)
def forward(self, src, src_len):
    outputs, hidden = self.gru(emb)     # [B,S,2H]、[2,B,H]
    fwd, bwd = hidden[0], hidden[1]
    return outputs, torch.cat([fwd, bwd], -1)   # (all states, last 2H)
```

### Step 2: `Bridge` 初始化 Decoder

```python
bridge = Bridge(2 * HID, HID)
h0 = bridge(enc_last)    # [1,B,H]
```

### Step 3: 装配 `BiSeq2Seq` + `__main__` 自检

1. **双向 shape**：`outputs [B,S,2H]`、`last [B,2H]` → `[PASS]`
2. **Bridge 装配**：`2H→H`，Decoder 单步 `logits [B,V]`、`hidden [1,B,H]` → `[PASS]`
3. **教师强制收敛**：loss **1.275 → 0.291** → `[PASS]`

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **双向拼接 `[fwd;bwd]` 而非相加/平均** | 保留两路独立的语义；也让 Attention 的 Key/Value 信息（v0.3）一起翻倍，更完整。 |
| **Bridge 用 `tanh`** | LSTM/GRU 的隐状态范围在 `(-1,1)`，`tanh` 让初始状态落回同一量纲，避免数值漂移。 |
| **终态单向 `最后位置` 即可，无需所有位置** | 双向 RNN 的终态 `hidden[-1]` 已经聚合了整条序列（前后双向跑完），与 v0.1 `c=h_n` 的取法一致。 |

## Going Deeper

- **双向不适合严格流式解码**：要看 `<后面` 的词才能定当前位置，翻译在线逐字出结果时就做不到。所以 streaming 场景保留单向；双向常见于离线翻译/摘要/序列标注。
- **计算量约为单向的 2×**：两个方向各自跑一遍 GRU。
- **Bridge 在深层/带注意力时更复杂**：m06(v0.5 深层) 会发现多层 Encoder 与多层 Decoder 的 Hidden/Context 桥接不是一行 Linear；m04 的 Attention 解码器配合双向时会多一步「用那个 2H 的 Encoder 输出做 attention」。

## 模块定位

m05 让 Encoder 从"单向看左"进化为"左右都看"，并引入 Bridge 把 `2H` 状态喂给 Decoder。m06（v0.5）将在 Encoder/Decoder 内部加深层数、加 Dropout / Residual / LayerNorm，继续提升容量与泛化。