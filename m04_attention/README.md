# m04: Bahdanau Attention — 让 Decoder 每步重新回顾输入（v0.3）

[返回根目录](../README.md)

---

## The Problem

m02 的 MVP 用**一个固定向量** `c = h_n` 初始化 Decoder——整条输入被压进最后一步的隐状态。这有两个致命问题（`versions.md` 第一节点得很清楚）：

- 输入越长，越靠前的信息越容易被末尾挤掉（长依赖变差）；
- 生成"三"的时候，本应只盯着输入里的 `three`，却被迫「平等地」看整条 `three one nine`，`一二三` 一多，模型分不清该抄哪个位置。

**v0.3 的目标**：让 Decoder 在生成每个输出词时，**动态地**重新审视整条 Encoder 输出，按相关性分配权重——这就是 Attention。

## The Solution

Encoder 不再只吐出最后一个隐状态，而是把**所有位置的隐状态 `h_1..h_n` 全部保留**。Decoder 在每步用一个 **Bahdanau Attention** 层：

```
对齐分数  e_{t,i} = vᵀ tanh(W_h h_i + W_s s_{t-1})
权重      α_{t,i} = softmax_i( e_{t,i} )        （对源位置归一化）
上下文    c_t     = Σ_i α_{t,i} h_i             （按权重加权整条输入）
Decoder 输入 = [E(y_{t-1}) ; c_t]                （拼上一词的 embedding）
```

每个 Decoder 位置 `t` 都有**自己的** context 向量 `c_t`，突破"固定向量"瓶颈。

## How It Works

### 1. Encoder 返回全部状态

```python
def forward(self, src, mask):
    outputs, hidden = self.gru(emb)   # outputs [B,S,H]、hidden [1,B,H]
    return outputs, hidden.squeeze(0)
```

`outputs`（每个源位置的隐状态）交给 Attention 作为 Key/Value；`hidden` 作为 Decoder 的初始状态。

### 2. BahdanauAttention：投影 → 能量 → softmax → 加权和

```python
enc_f = self.enc_proj(enc_outputs)              # [B,S,A]  源位置投影
dec_f = self.dec_proj(dec_hidden).unsqueeze(1)  # [B,1,A]  解码状态投影
energy = torch.tanh(enc_f + dec_f)            # [B,S,A]
e = self.score(energy).squeeze(-1)            # [B,S]     对齐分数
e = e.masked_fill(~src_mask, float('-inf'))   # 屏蔽 PAD 位置
alpha = F.softmax(e, dim=-1)                  # [B,S]     权重
ctx = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)  # 加权和
```

Mask 让 PAD 位置进不了对齐（`softmax(-inf)=0`）——这与 m03 的 v0.2 一致，但这次屏蔽的是"源序列的 PAD"。

### 3. Decoder：拼接 embedding 与 context 一起喂

```python
emb = self.embed(input_token)                      # [B, E]
emb = torch.cat([emb, ctx], dim=-1).unsqueeze(1)   # [B, 1, E+H]
out, hidden = self.gru(emb, hidden)                # 输入维度升到 E+H
logits = self.out(out.squeeze(1))
```

## Code Walkthrough

> 下面每行都对应 `code.py` 里的真实实现。

### Step 1: `BahdanauAttention.forward`

```python
ctx, alpha = self.attention(dec_hidden_last, enc_outputs, src_mask)
```

### Step 2: Seq2Seq 主 forward（v0.3 相比 v0.1 的唯一结构变化）

```python
enc_out, enc_hidden = self.enc(src, mask)          # 不再只取最后隐状态
hidden = enc_hidden.unsqueeze(0)
for t in range(1, tgt.size(1)):
    out, hidden, alpha = self.dec(dec_in, hidden, enc_out, mask)
    # dec 内部：ctx, alpha = attention(...) ; 输入=[emb;ctx]
    dec_in = ...
```

### Step 3: `__main__` 自检（v0.3，单元级 + 收敛）

1. **注意力数学正确**（确定性，不依赖训练）：
   - 有效位置权重`sum == 1`（softmax 归一）；
   - 被 mask 的位置权重 `== 0`（`-inf`）；
   - `context == hand-compute(Σ α_i h_i)`（加权和恒等）→ `[PASS]`
2. **装配 shape 正确**：Encoder→Attention→Decoder 全链路跑通 → `[PASS]`
3. **教师强制下收敛**：loss **1.006 → 0.012** → `[PASS]`

（greedy 解码质量属于 v0.6 Beam / v0.7 Scheduled Sampling 的范畴，本模块仅打印供观察，不做断言——见「Going Deeper」。）

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **v0.3 采单元级数学断言而非端到端翻译** | 端到端贪心解码受 Exposure Bias 干扰（训练喂真值、推理喂自己），那正是 v0.6/v0.7 的战场。v0.3 先把手写的注意力机制用**确定性的数学恒等式**（mask→0、softmax→1、ctx=加权和）证明正确。 |
| **context 拼进 Decoder 输入而非输出投影** | 采纳 Bahdanau（`s_t = GRU([E(y);c_t], s_{t-1})`）原版，与 `versions.md` 的公式完全一致；Luong 做法则把 c 拼到输出前面，两者在性能上差异有限。 |
| **mask 用 `-inf` 而非归零** | 与 v0.3 之前的做法一致：`softmax(-inf)=0` 保证被屏蔽位置权重恒为 0 且重新归一化。 |

## Going Deeper

- 如果你运行 `python m04_attention/code.py`，会看到 greedy 输出像 `一 三 九 九 一`——它从第一个 token 起就走偏。这不是 Attention 坏了，而是**训练用的 Teacher Forcing 与推理用的自回归解码分布不一致**（Exposure Bias）：训练时模型见过真标签、推理时只见过自己的预测。`versions.md` 明确把解决它放在 v0.7（Scheduled Sampling / Label Smoothing），当前 v0.3 只负责"把 Attention 做对"。
- v0.7 之后配上 v0.6 的 Beam Search，这个 toy 的 greedy 也会收敛到 `three one nine → 三 一 九`。
- attention 权重即"对齐矩阵"(alignment)：它可以可视化哪两个词对齐——这也是 Attention Seq2Seq 早期被称为 `alignment` 模型的由来。

## 模块定位

m04 实现了 Seq2Seq 历史上**最重要的一次升级**——从固定 Context 到动态 Attention。你用单元级断言证明了机制本身正确，同时诚实地点明"端到端解码还需要 v0.6/v0.7"。接下来 m05 在 Encoder 上再加双向（v0.4），让每个源位置同时看到左右上下文。