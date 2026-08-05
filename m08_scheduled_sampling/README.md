# m08: Scheduled Sampling + Label Smoothing — 对齐训练/推理、降过度自信（v0.7）

[返回根目录](../README.md)

---

## The Problem

`versions.md` 第 7 节点出两个 Seq2Seq 的老毛病：

**问题一：Exposure Bias。** 训练时用 Teacher Forcing（把**真值**喂回 Decoder），推理时却用**模型自己的预测**。两种分布不一致——训练得很好的模型，一到推理自回归就可能因一步偏差而崩坏。m04 里 greedy 输出 `三 九 一`（顺序乱、不吐 EOS）正是它的现场直播。

**问题二：过度自信（overconfidence）。** 纯 one-hot 的 Cross Entropy 会逼模型把金标概率推到 1。对噪声标签或小数据，这往往造成过拟合、越自信越错。

## The Solution

两个互补机制：

```
Scheduled Sampling：训练时以 p(epoch) 概率喂真值，p 随时代指数衰减
   "先喂真值把模型教会，再逐步逼它吃自己吐的，消除训练/推理分布差"

Label Smoothing：把 one-hot 目标抹平
   金标质量 = 1-ε+ε/V，其余每类 = ε/V   ← 阻止 model 追求绝对自信
```

## How It Works

### 1. Scheduled Sampling 调度

概率随 epoch 指数衰减、且有下界：

```
ratio(epoch) = max(minimum, initial · decay^(epoch-1))
epoch:  1    2    3    4    5    6    7    8
ratio:  1.0 → 0.8 → 0.64 → 0.51 → 0.41 → 0.33 → 0.26 → 0.21
```

### 2. Label Smoothing 的定义（PyTorch 约定）

PyTorch 把 `ε` 均匀分摊在**全部 V 类**（含金标）上：

```
L = (1-ε)·(-log p[g]) + ε·(1/V)·Σ_{c=1..V} (-log p[c])
```

> 容易误区：不是 `ε/(V-1)` 只摊给非金标类——PyTorch 的 `label_smoothing` 连金标一起摊。这在自检 [B] 里用手算展开式对照验证。

## Code Walkthrough

> 下面每行都对应 `code.py` 里的真实实现。

### Step 1: Scheduled Sampling 调度（v0.7）

```python
def scheduled_ratio(epoch, initial=1.0, minimum=0.0, decay=0.9):
    r = initial * (decay ** (epoch - 1))
    return max(minimum, r)
```

训练里每个 epoch 用 `tf = scheduled_ratio(epoch)`，`F.cross_entropy(..., label_smoothing=0.1)`。

### Step 2: `__main__` 自检

1. **调度单调递减且有界**：`[1.0, 0.8, 0.64, 0.51, ...]` → `[PASS]`
2. **Label Smoothing 公式恒等**：`nn.CrossEntropyLoss(label_smoothing=ε)` == 手算展开式 → `[PASS]`
3. **平滑目标合法**：每类 `ε/V`、金标 `1-ε+ε/V`，总质量 == 1 → `[PASS]`
4. **配平滑+调度采样训练收敛**：loss **1.592 → 0.687** 且全程有限 → `[PASS]`

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **Scheduled Sampling 用指数衰减** | 简单、有单调性，是本系列从 m02 就同步出现的"逐步降 Teacher Forcing"的正式化。注意 versions.md 也提醒：它并非处处更好，应该用验证集比较，不默认一定提升。 |
| **校验选"机制恒等"而非"端到端翻译"** | 端到端贪心质量受 Exposure Bias + Beam 参数共同影响，会掩盖 v0.7 到底修了什么。`label_smoothing` 的**展开式恒等**是确定性、可复现、能一眼看懂的机制证明。 |
| **`eps/V` 而非 `eps/(V-1)`** | 这是 PyTorch 的实际约定，也是初学者最容易写错的点；README 与自检一起钉死。 |

## Going Deeper

- **Label Smoothing 的不利面**：它会让模型"不那么自信"，在需要高置信度给用户的生成任务里可能让输出偏模糊；对小词表/关键是"绝对精确"的任务（如日期格式）酌情。`ε=0.1` 是常用默认，可通过验证调。
- **Scheduled Sampling 的权衡**：它把"训练目标"从"预测下一个真 token"变成"预测自己也能常用的 token"——收敛可能更慢、目标更不一致；如果验证集收益不明显，用纯 Teacher Forcing + 更多 Beam 也常等于效果。
- **v0.7 完成后** greedy 的稳定性已经有保障（不再依赖真值），这让 v0.8 转向真实数据的评估变得有意义。

## 模块定位

v0.7 补上了"训练侧"的两处修：Exposure Bias（Scheduled Sampling）与过度自信（Label Smoothing），并把它们的机制用恒等式钉死。m09（v0.8）开始把 toy 数据换成 Subword + 真实评测（BLEU），迈向真实 NLP。