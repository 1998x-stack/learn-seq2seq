# m01: 手写 RNN / LSTM / GRU Cell — 从"一条隐状态"到"门控记忆"（理论段）

[返回根目录](../README.md)

---

## The Problem

Seq2Seq 想用神经网络处理序列，第一步是让网络**有"记忆"**——读到第 5 个词时，还能记着第 1 个词留下的信息。`versions.md` 的第一、二、三节把这条线拆成三个递进问题：

- **问题一（Vanilla RNN）**：如何让一个网络"按时间递归"地更新状态？最朴素的做法：`h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`，一条隐状态通路反着往前走。但这条通路有个致命伤——反向传播要连续乘很多个 $\partial h_t/\partial h_{t-1}$，特征值 < 1 时梯度指数级衰减（**梯度消失**），学不动长程依赖。
- **问题二（LSTM）**：怎么让梯度在时间上"抄近路"？LSTM 引入一条独立的 **Cell State**，在上面做加性更新 `c_t = f_t*c_{t-1} + i_t*g_t`——加法的梯度近乎恒等传递，缓解了消失。
- **问题三（GRU）**：LSTM 要 4 个门、2 个状态，能不能更省？GRU 把 c/h 合并成单一 h、只留 2 个门，参数更少、速度更快，作为实现 Seq2Seq MVP 的默认选择。

用户 (学习者) 需要的不是"会用 `nn.LSTM`"，而是**亲手写出每个 cell 的张量数学、看懂每个门在干什么、并验证它和 PyTorch 内置实现一致**。

## The Solution

**手写三种 Cell**，全部用显式参数 + 纯矩阵运算，不碰 `nn.RNN / nn.LSTM / nn.GRU`：

```
RNN  :  h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)      —— 一条隐状态通路
LSTM :  i/f/g/o 四个门 + c_t 加性路径                  —— 缓解梯度消失
GRU  :  r/z 两个门 + 候选状态 n + (1-z)n + z·h        —— 更简、参数更少
```

三种 Cell 都落在同一个"手写 vs PyTorch 内置"的对齐断言上：喂同一份权重，单步 forward 结果最大绝对误差 **5.96e-08**（仅浮点舍入），说明公式语义与官方实现完全一致。

## How It Works

### 1. Vanilla RNN：一条隐状态，梯度随深度消失

```
h_t = tanh( W_ih x_t + b_ih + W_hh h_{t-1} + b_hh )
```

反向传播时从第 T 步回传到第 t 步，要连乘 `∏_{k=t+1}^T ∂h_k/∂h_{k-1}`。当这一串矩阵的特征值 < 1，乘积指数级趋近 0——这就是为什么朴素 RNN 记不住远距离信息。

### 2. LSTM：加一条 Cell State 给梯度"抄近路"

在 LSTM 里，三个门和一个候选从同一份 `gates` 张量切分出来：

| 符号 | 名称 | 作用 | 激活 |
|---|---|---|---|
| `i` | 输入门 | 决定写入多少新信息 | sigmoid |
| `f` | 遗忘门 | 决定保留多少旧记忆 | sigmoid |
| `g` | 候选记忆 | 新写入的内容 | tanh |
| `o` | 输出门 | 决定暴露多少 cell state | sigmoid |

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     § 加性路径：梯度近乎直通
h_t = o_t ⊙ tanh(c_t)
```

### 3. GRU：合并状态，两个门

GRU 把 c/h 合并成单一 h，权重布局 `[r | z | n]`（与 PyTorch `weight_ih` 一致）：

```
r_t = σ(W_ir x_t + W_hr h_{t-1})    重置门：候选读多少旧状态
z_t = σ(W_iz x_t + W_hz h_{t-1})    更新门：保留多少旧状态
n_t = tanh(W_in x_t + r_t ⊙ (W_hn h_{t-1} + b_hn))   候选状态（先被重置门加权）
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
```

> 候选状态的坑：`r_t` 要「先关联进旧状态再进 `tanh`」，所以它不能像 RNN 那行直接 `x@W + h@W`。我们分块取权重 `W_ih[2H:]`（第 3 块是 `n`），保留 autograd，不需要 detach。

## Code Walkthrough

> 下面每行都对应 `code.py` 里的真实实现。核心是三个 Cell 的 `forward` + 一个统一「时间展开」工具 `unfold`。

### Step 1: RNN Cell（理论段）

```python
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()   # xavier_uniform

    def forward(self, x_t, h_prev):
        return torch.tanh(
            x_t @ self.W_ih.t() + self.b_ih
            + h_prev @ self.W_hh.t() + self.b_hh
        )
```

### Step 2: LSTM Cell（四门一份 `gates`）

```python
gates = x_t @ self.W_ih.t() + self.b_ih + h_prev @ self.W_hh.t() + self.b_hh  # [B,4H]
i, f, g, o = gates.chunk(4, dim=-1)
c_t = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
h_t = torch.sigmoid(o) * torch.tanh(c_t)
```

### Step 3: GRU Cell（三门分块取 `n`，保留梯度）

```python
t = x_t @ self.W_ih.t() + self.b_ih + h_prev @ self.W_hh.t() + self.b_hh     # [B,3H]
r, z, n = t.chunk(3, dim=-1)
H = self.hidden_size
W_in, b_in = self.W_ih[2*H:], self.b_ih[2*H:]
W_hn, b_hn = self.W_hh[2*H:], self.b_hh[2*H:]
n = torch.tanh(x_t @ W_in.t() + b_in + torch.sigmoid(r) * (h_prev @ W_hn.t() + b_hn))
return (1 - torch.sigmoid(z)) * n + torch.sigmoid(z) * h_prev
```

### Step 4: `__main__` 自检

1. **数值对齐**：把同一份权重拷贝进手写 Cell 与 `nn.RNNCell/LSTMCell/GRUCell`，单步输出 `max abs diff == 5.96e-08` → 打印 `[PASS] forward == PyTorch`
1.5. **反向正确性**：解析梯度 vs 中心有限差分，三种 Cell 全部一致 → 打印 `[PASS] grad == finite-difference`
2. **能学会时间记忆**：`unfold` 逐 step 展开，训练一个"延时一位复现"任务——第 t 步预测第 t-1 步的 token。三种 Cell 的 loss 都从 ~2.0 掉到 ~0.2 → 打印 `[PASS] loss decreased`

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **权重用 `nn.Parameter` 而非 `nn.Linear` 组合** | 教学要暴露每个 gate 的矩阵语义；`nn.Linear` 隐藏了维度拼接。手写让「r/z/n 各自是第几块」一目了然。 |
| **GRU 候选单独分块算** | `r·h_{t-1}` 必须先进 `tanh`；把 `n` 块权重 `W_ih[2H:]` 单独切片是对的公式，同时自然保留 autograd（不 detach）。 |
| **与 PyTorch 内置做数值对齐** | 这是"手写是否正确"的黄金基准。教科书公式的 `sigmoid/tanh`、门序、拼接顺序都可能写反；对齐断言把这些一次性捕获。 |
| **用中心有限差分校验梯度** | `forward==PyTorch` 只证**前向**对；反向（BPTT）同样容易错（`c_t=f*c+i*g` 的乘积与门求导）。加一条"解析梯度==数值梯度"的自检，才能真正兜住反向。这也暴露了 torch 2.10 的 `nn.LSTMCell` 在 CUDA+double 下的反向怪癖——本模块的 LSTM 用差分校验通过，不依赖 torch。 |
| **张量放 device** | 教学代码片段容易把 `torch.zeros` 留在 CPU 而输入在 CUDA。统一用 `xs.device` 对齐所有临时张量。 |

## Going Deeper

- **为什么 GRU 参数更少**：粗略 `RNN:GRU:LSTM ≈ 1× : 3× : 4×`（每次变换数量）。所以 GRU 常是实现 Seq2Seq MVP 的好选择：比 RNN 稳、比 LSTM 省。
- **LSTM 的遗忘门初始化为偏大**（`bias` 起始取较大正值，让初始 `f≈1`）能显著提升长程记忆——工业生产常见技巧，本模块从零出发用 `bias=0`。
- **梯度消失 vs 梯度爆炸**：特征值 > 1 时 RNN 梯度爆炸，需 `clip_grad_norm`——这在本系列 m02 的 Seq2Seq 循环网络里会用到。
- **手写 += 内置**：`unfold` 逐 step 循环在 GPU 上固然慢，但它是「每个时间步都看到可打印中间状态」的教学形式——真正的生产实现会用 `torch.nn.GRU` 一次算整条序列。

## 模块定位

m01 是进入 Seq2Seq 之前的**地基**：它让你亲手摸到"一条隐状态"如何递归、"门控/cell state"如何救活梯度、GRU 如何更省。m02 就把这些 GRU 作为 Encoder 和 Decoder 的组件，拼出第一个可翻译数字序列的 MVP（v0.1）。