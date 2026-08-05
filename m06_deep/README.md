# m06: 深层 Seq2Seq — Multi-layer + Dropout + Residual + LayerNorm（v0.5）

[返回根目录](../README.md)

---

## The Problem

m02–m05 都是**单层** GRU。把层多加是模型"容量"的主要来源，但深层循环网络有一串经典病症（`versions.md` 第 12 节点明）：

- **梯度传播更困难**：多层 GRU 叠加后，反向传播的路径更长；
- **Dropout 陷阱**：RNN 的 `dropout` 只作用于**层与层之间**，`num_layers=1` 根本不生效；
- **过拟合小型数据**：参数越多，越容易把训练集背下来；
- **数值不稳**：深层堆叠后的隐状态分布可能漂移（NaN / 爆炸）。

v0.5 的目标：把 Encoder/Decoder 加深到 **3 层**，同时配齐 Dropout、Residual Connection、Layer Normalization、梯度裁剪、学习率衰减，让"更深的模型"既能训得动、又不过度崩坏。

## The Solution

```
Single-layer GRU
     ↓
Multi-layer GRU(3)        Encoder / Decoder 内部层叠
     +
Dropout(0.2)              层间正则
     +
Residual（decoder：h + 输入投影）  残差短路，缓解深层梯度
     +
LayerNorm                 稳定隐状态分布
     +
Gradient Clipping(2.0)    防爆炸
```

教师式训练下 loss 应从 ~1.3 收敛到 <0.1 且**全程有限（无 NaN）**。

## How It Works

### 1. `nn.GRU(num_layers=3, dropout=0.2)`

`num_layers=3` 让 GRU 内部堆三层；`dropout=0.2` 只在**第 1→2、2→3 层之间**加，末层不加。关键坑：**`num_layers=1` 时 `dropout` 参数被直接忽略**——很多人设了 dropout 却没生效。

### 2. Multi-layer 的隐状态是 3-D

单层 GRU 的 hidden 是 `[1, B, H]`，多层是 `[L, B, H]`——每一层一个终态。Decoder 的初始状态必须整个 `[L,B,H]` 一起给（**不能只给顶层 2-D**），否则报 "hx should be 3-D"。

```python
hidden = enc(src, lens)[1]   # [L, B, H]
h = hidden                   # 整个多层状态传给 decoder
out, next_hidden = dec(dec_in, h)
```

### 3. Residual + LayerNorm（Post-Norm）

Decoder 单步里，把输入 embedding 投影到 H 后**加**到 GRU 输出上（真实残差短路），再做 LayerNorm：

```python
emb = self.embed(token)                 # [B, E]
h = self.gru(emb.unsqueeze(1), h)[0].squeeze(1)   # GRU 输出 [B,H]
h = h + self.res_proj(emb)              # 残差：输入投影加到输出
out = self.out(self.ln(h))              # LayerNorm 再投影到词表
```

> 为什么先投影：embedding 维度 E(64) ≠ 隐状态 H(128)，残差要等维，所以先 `res_proj: E→H`。

## Code Walkthrough

### Step 1: 深层 Encoder（v0.5）

```python
self.gru = nn.GRU(embed, hid, num_layers=3, dropout=0.2, batch_first=True)
def forward(self, src, src_len):
    outputs, hidden = self.gru(emb)    # [B,S,H]、[3,B,H]
    return outputs, hidden             # 所有层终态都保留
```

### Step 2: 深层 Decoder（Residual + LayerNorm）

```python
h = h + self.res_proj(emb)            # 残差短路
return self.out(self.ln(h)), next_hidden
```

### Step 3: `__main__` 自检

1. 多层 Encoder：`hidden [3,B,H]`、`outputs [B,S,H]` → `[PASS]`
2. 多层 Decoder 初始化（3-D hidden）+ 单步 logits shape → `[PASS]`
3. 深层模型训练：loss **1.307 → 0.069** 且全程有限 → `[PASS]`

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **多层 `hidden` 必须 3-D 全程传递** | 这是 `nn.GRU(num_layers>1)` 的硬性 API 要求：初始状态要么不传（默认全零），要么给 `[L,B,H]`。传给 Decoder 时 2-D 直接报错。 |
| **Residual 放在输入投影上** | 让"残差 + 输入"作为一条可学习短路，既有残差的梯度收益，又不需要额外状态；用 Post-Norm（`+` 后 LN）对应现代 Transformer 的常见顺序。 |
| **`nn.GRU(dropout=...)` 只在多层生效** | 这个"陷阱"本身就是教学点：单层模型设 dropout 是摆设。 |
| **梯度裁剪阈值从 1.0 提到 2.0** | 深层允许稍微大一点的梯度进行而有意义；同时靠 LayerNorm 兜底数值稳定性。 |

## Going Deeper

- **节目层数 vs 隐维度**：层数、hidden 都加一倍，参数约 4×。坏 case：过拟合比欠拟合快得多，所以 Dropout + Early Stop 几乎是标配。
- **Bridge 在深层里更绕**：m04 里 Bridge 只是 `Linear(2H→H)tanh`；多层 Encoder+Decoder 时往往需要 `nn.Linear(D*L, D)` 或用 Encoder 的 top-layer state 直接初始化 Decoder 顶层，其余层用零填充。
- **Residual 的两种顺序**：`Post-Norm（+ 后 LN）`（本模块）vs `Pre-Norm`（先 LN 再加）。Pre-Norm 在超大模型上更稳。真正的现代解码块（含 Transformer）在 m12 会用到。

## 模块定位

v0.5 让 Seq2Seq 从"单层玩具"升级到"多层、带正则、稳定收敛"，同时踩掉了 `num_layers=1` 的 dropout 陷阱与 3-D hidden 的坑。m07（v0.6）开始换解码器：从 Greedy 到 **Beam Search**，直接提升推理质量。