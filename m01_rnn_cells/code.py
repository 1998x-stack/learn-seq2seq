"""m01 RNN / LSTM / GRU Cells — 手写三种循环单元，理解"时间递归"与"门控"。

Run:  python m01_rnn_cells/code.py

本模块手写实现 Vanilla RNN、LSTM、GRU 三种循环 Cell：
  - RNN  : h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)               单隐状态递归
  - LSTM : 引入独立的 Cell State c_t 与 输入/遗忘/输出 三个门       缓解梯度消失
  - GRU  : 合并状态，（更新门 + 重置门）两个门                   更简、参数更少

每种 Cell：
  1) 用纯矩阵乘法手写前向，完全不调用 nn.RNN / nn.LSTM / nn.GRU；
  2) 与 PyTorch 内置实现做数值对齐（同一份权重逐逐步对比），作为正确性基线；
  3) 在一个"延时一位复现"小任务上训练，验证三种单元都能学会时间记忆。
"""

from __future__ import annotations

import sys

# 防止 `import code` / `import pdb` 被本文件遮蔽（见 learn-llm 同款注释）。
import code as _stdlib_code  # noqa: E402
import pdb as _stdlib_pdb  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)
sys.modules.setdefault("pdb", _stdlib_pdb)

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 1. Vanilla RNN Cell（单隐状态递归）
# ----------------------------------------------------------------------------

class RNNCell(nn.Module):
    """h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)。

    只有一条隐状态通路，反向传播要连续乘很多个 d(h_t)/d(h_{t-1})；
    特征值 < 1 时梯度指数级衰减（梯度消失），学不动长程依赖。
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            x_t @ self.W_ih.t() + self.b_ih
            + h_prev @ self.W_hh.t() + self.b_hh
        )


# ----------------------------------------------------------------------------
# 2. LSTM Cell（Cell State + 三门）
# ----------------------------------------------------------------------------

class LSTMCell(nn.Module):
    """标准 LSTM。Cell State 上有直接加性路径（f*c_prev + i*g），
    梯度在 c 上近乎恒等传递，这是它缓解梯度消失的关键。
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 一次算 4 个门，服务 i(输入)/f(遗忘)/g(候选)/o(输出)
        self.W_ih = nn.Parameter(torch.zeros(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.zeros(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)

    def forward(self, x_t: torch.Tensor, state):
        h_prev, c_prev = state  # 各自 [B, hidden]
        gates = (
            x_t @ self.W_ih.t() + self.b_ih
            + h_prev @ self.W_hh.t() + self.b_hh
        )  # [B, 4*hidden]
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_t = f * c_prev + i * g      # 加性路径：梯度近乎直通
        h_t = o * torch.tanh(c_t)
        return h_t, (h_t, c_t)


# ----------------------------------------------------------------------------
# 3. GRU Cell（更新门 + 重置门，单状态）
# ----------------------------------------------------------------------------

class GRUCell(nn.Module):
    """将 c/h 合并为单一 h，只保留两个门：更新门 z、重置门 r。

    r_t = σ(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)   重置门
    z_t = σ(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)   更新门
    n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))   候选状态
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}            最终状态
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 3 块拼接：[r | z | n]，与 PyTorch GRUCell 布局（reset/update/new）一致。
        self.W_ih = nn.Parameter(torch.zeros(3 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.zeros(3 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        H = self.hidden_size
        t = x_t @ self.W_ih.t() + self.b_ih + h_prev @ self.W_hh.t() + self.b_hh
        r, z, n = t.chunk(3, dim=-1)            # [r | z | n]
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)

        # 候选状态需要"重置门先加权旧状态再进 tanh"，
        # 分块权重保留 autograd，无需任何 detach。
        W_in, b_in = self.W_ih[2 * H:], self.b_ih[2 * H:]
        W_hn, b_hn = self.W_hh[2 * H:], self.b_hh[2 * H:]
        pre = x_t @ W_in.t() + b_in + r * (h_prev @ W_hn.t() + b_hn)
        n = torch.tanh(pre)

        return (1 - z) * n + z * h_prev


# ----------------------------------------------------------------------------
# 4. 统一"按时间展开 + one-hot"的小型工具（供训练/对比使用）
# ----------------------------------------------------------------------------

def unfold(cell, x_ids: torch.Tensor, h0):
    """把 [B, T] 的整数序列 one-hot 后逐时间步送入 cell，返回每步隐状态。

    返回 outs: [B, T, hidden]；LSTM 用 (h, c)，其余用 h。
    """
    B, T = x_ids.shape
    vocab = cell.input_size
    is_lstm = isinstance(cell, LSTMCell)
    outs = []
    if is_lstm:
        h, c = h0, torch.zeros_like(h0)
    else:
        h = h0
    for t in range(T):
        x_t = F.one_hot(x_ids[:, t], num_classes=vocab).float()
        if is_lstm:
            h, (h, c) = cell(x_t, (h, c))
        else:
            h = cell(x_t, h)
        outs.append(h)
    return torch.stack(outs, dim=1)  # [B, T, hidden]


def make_delay_task(B: int, L: int, vocab: int) -> torch.Tensor:
    """生成 token 序列；自回归目标 = 第 t 步复现第 t-1 步的 token。"""
    return torch.randint(0, vocab, (B, L)).to(device)


# ----------------------------------------------------------------------------
# 5. 主程序：自检 + 数值对齐 + 学习小任务
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)

    torch.manual_seed(0)

    # ---- [1] 与 PyTorch 内置实现的单步数值对齐 ----
    pairs = [
        (RNNCell, nn.RNNCell),
        (LSTMCell, nn.LSTMCell),
        (GRUCell, nn.GRUCell),
    ]
    torch.manual_seed(0)
    for mine_cls, torch_cls in pairs:
        mine = mine_cls(4, 16)
        ref = torch_cls(4, 16)
        # 拷贝同一份权重，确保只对比"手写公式 vs 内置"而非随机初始化。
        mine.W_ih.data.copy_(ref.weight_ih.data)
        mine.W_hh.data.copy_(ref.weight_hh.data)
        mine.b_ih.data.copy_(ref.bias_ih.data)
        mine.b_hh.data.copy_(ref.bias_hh.data)
        x = torch.randn(5, 4)
        h0 = torch.randn(5, 16)
        if isinstance(mine, LSTMCell):
            c0 = torch.randn(5, 16)
            h_m, _ = mine(x, (h0, c0))
            h_r, _ = ref(x, (h0, c0))
        else:
            h_m = mine(x, h0)
            h_r = ref(x, h0)
        diff = (h_m - h_r).abs().max().item()
        assert diff < 1e-5, (mine_cls.__name__, diff)
        print(f"[PASS] {mine_cls.__name__} forward == PyTorch (max abs diff {diff:.2e})")

    # ---- [2] 三种 Cell 都能在"延时一位复现"任务上学到时间记忆 ----
    # 自监督：第 t 步预测第 t-1 步的 token，强制 cell 在时间上保留一位信息。
    print("---- 延时一位复现记忆任务（loss 应从高到低） ----")
    B, L, vocab, H = 32, 24, 5, 20
    xs = make_delay_task(B, L, vocab)
    y = xs[:, :-1].reshape(-1)
    for name, cls in (("RNN", RNNCell), ("LSTM", LSTMCell), ("GRU", GRUCell)):
        cell = cls(vocab, H).to(xs.device)
        readout = nn.Linear(H, vocab).to(xs.device)
        nn.init.normal_(readout.weight, std=0.5)
        readout.bias.data.zero_()
        optim = torch.optim.Adam(list(cell.parameters()) + list(readout.parameters()), lr=3e-3)
        losses = []
        for _ in range(80):
            h0 = torch.zeros(B, H, device=xs.device)
            outs = unfold(cell, xs, h0)[:, :-1]     # [B, L-1, H]
            logits = readout(outs)                    # [B, L-1, vocab]
            loss = F.cross_entropy(logits.reshape(-1, vocab), y)
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(float(loss.item()))
        fin = losses[-1]; init = losses[0]
        print(f"  {name}: loss {init:.3f} -> {fin:.3f} (ratio {fin/init:.3f})")
        assert fin < init, f"{name} did not learn"
        print(f"    [PASS] {name} loss decreased")

    print("\n全部自检通过。手动实现继承 autograd，可正常反向传播。")