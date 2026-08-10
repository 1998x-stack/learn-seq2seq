[返回根目录](../README.md)

# m12: Transformer Encoder-Decoder — 拆除循环依赖、并行训练（v1.1）


---

## The Problem

m04 的 Attention Seq2Seq 让 Decoder **动态回顾**整条 Encoder 输出，但 Encoder 和 Decoder **内部仍是 RNN、顺序计算**：

```
h_t = f(h_{t-1}, x_t)      ← 第 2 个词必须等第 1 个词算完
```

`versions.md` 的 v1.1 点出三个后果：
1. **无法在时间维并行**：GPU 每次都串行走完整条序列；
2. **长距离路径太长**：相距很远的两个 token 要跨很多递归步才能交互；
3. **BPTT 展开很深**：梯度经多次相乘易消失/爆炸。

## The Solution

Transformer **完全移除循环**，Encoder 和 Decoder 内部全部改用**自注意力**（Self-Attention）：一条序列的所有位置**同时**彼此交互。按发展关系：

```
Seq2Seq：固定向量承载全部输入信息
   ↓
Attention Seq2Seq：Decoder 动态访问 Encoder 全部状态（但仍顺序）
   ↓
Transformer：Encoder/Decoder 内部也全用 Attention，位置并行
```

结构（小规模：`D=64, heads=4, layers=2`）：

```
Encoder: Embedding → PosEnc → N×(SelfAttn + FFN + Res + LN)
Decoder: Embedding → PosEnc → N×(SelfAttn[因果] + CrossAttn + FFN + Res + LN)
Head   : Linear → 目标词表
```

## How It Works

### 1. 自注意力：一图全并行

每个位置同时是 Q（查询）/K（钥匙）/V（内容）。`F.scaled_dot_product_attention` 一次算完整 `[B×H×T×T]` 的张量——这就是"并行"的由来：**没有 for 时间步**。自检 [A] 断言 Encoder 单次前向直接得到 `[B,S,H]`。

### 2. 因果掩码（Decoder Self-Attention）

Decoder 的 `t` 位只能看 `≤t`，否则答案泄漏。掩码 = `torch.triu(full(T,T,-inf), diagonal=1)`；`softmax(-inf)=0`，未来位置权重恒为 0。自检 [B] 精确断言上三角权重 ≈0。

### 3. Cross-Attention（对话两端）

Decoder 的 Q 只来自自身，K/V 来自 **Encoder 的 memory**；用 padding 掩码把源端 `<PAD>` 位置置 `-inf`。自检 [C] 验证整条 Decoder 前向 shape。

### 4. Positional Encoding

自注意力没有先后概念，用正弦位置编码把位置加进去：

```python
pe[:, 0::2] = sin(pos / 10000^(2i/d)) ; pe[:, 1::2] = cos(...)
x = x + self.pe[:, :x.size(1)]
```

## Code Walkthrough

### Step 1: 多头注意力（v1.1 核心）

```python
q = self._reshape(self.q(query)); k = ...; v = ...
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)   # 并行一次算整张矩阵
out = out.transpose(1,2).contiguous().view(B, T, -1)
return self.out(out)
```

### Step 2: 关键坑——mask 用 `torch.where`，别用 `bool × -inf`

```python
key_mask = torch.where(padding_mask.unsqueeze(1).unsqueeze(2),
                       float("-inf"), 0.0)
```

> **为什么不能 `bool_tensor * -inf`**：`False(0) × -inf = NaN`（IEEE `0×∞`），会在"本应是 0 的真位置"注入 NaN。这也是本模块训练一开始 `loss=nan` 的元凶；改 `torch.where(bool, -inf, 0.0)` 后 8 个 epoch 从 **1.769 → 0.031**。

### Step 3: `__main__` 自检

`[A]` 并行 shape → `[PASS]`；`[B]` 因果掩码 → `[PASS]`；`[C]` CrossAttn shape → `[PASS]`；`[D]` 训练 `1.769→0.031` → `[PASS]`。

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **用 `F.scaled_dot_product_attention`** | 官方 fused 实现：内置 `1/√d` 缩放、支持 `-inf` 掩码、GPU 高效；比手搓 [B,H,T,T] 更快更稳，且仍是"显式注意力"教学语义。 |
| **Pre-Norm + Residual** | 先 LN 再子层，梯度更稳、超大模型默认（m06 已铺垫 Post/Pre-Norm 差异）。 |
| **`torch.where(bool,-inf,0)` 而非 `bool×-inf`** | 直接规避 `0×∞=NaN`——这是自注意力掩码最常见的 NaN 来源。 |
| **Encoder 不用因果、Decoder 要因果** | Encoder 读整段源（双向）；Decoder 每步只见已生成的 → 两类 mask 分开建。 |

## Going Deeper

- **复杂度**：自注意力是 `O(T²)`，长序列靠 v2.x 的 FlashAttention / 滑动窗口 / KV-Cache。
- **架构更细**：真正的 Transformer 还含 cross-attn 的 ln2 与 FFN 的 ln3 分开、embedding scale、学习率 warmup——这里把"最小可并行 Seq2Seq"讲透，生产细节留给 LLM 系列（`learn-llm`）。
- **从 RNN 到 Transformer 的主线**：本模块是 `versions.md` 十九节推荐顺序的收官——`RNN→LSTM/GRU→Seq2Seq→Attention→Transformer`，你全程手写见证。

## 模块定位

m12 完成 v1.1：把整条循环链路换成并行自注意力，拆掉循环依赖。至此 `versions.md` 的 **v0.1 → v1.1** 迭代路线全部落地——从一行 RNN Cell 到可并行训练的 Transformer，13 个模块闭环。继续沿 `versions.md` 向 Encoder-only（v3.x）/ Encoder–Decoder 变体（v4.x）前进时，本系列已为你备好全部底座。