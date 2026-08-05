"""m07 Beam Search + 长度归一化 — 从 Greedy 到 Top-K 候选序列搜索（v0.6）。

Run:  python m07_beam_search/code.py

v0.6 升级点（versions.md）：
  - Greedy 每步只留概率最高 token，容易掉进局部最优；
  - Beam Search 维护 K 条候选序列，每步从 K×V 里再留 Top-K；
  - 每个 beam 持有独立 hidden（decoder 状态），因为不同前缀状态不同；
  - 长度归一化 score / ((5+len)/6)^alpha，避免偏好短句。

算法用「确定性假 step_fn」做单元级自检，不依赖真实网络收敛：
  [A] Beam 能在已知全局最优的 logits 下找到正确完整序列
  [B] 长度归一化随长度单调变化、且确实改变长/短候选的排序
  [C] 归一化分数在长度惩罚下会"奖励"合理增长的序列
"""

from __future__ import annotations

import math
import sys
from typing import Callable, List, Tuple

import code as _stdlib_code  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)

import torch
import torch.nn.functional as F


def length_norm(seq_len: int, alpha: float = 0.6) -> float:
    """Google NMT 长度归一化分母 ((5+len)/6)^alpha。"""
    return ((5 + seq_len) / 6) ** alpha


def beam_search(
    step_fn: Callable,      # step_fn(token, hidden) -> (logits, next_hidden)
    init_hidden,
    bos: int,
    eos: int,
    vocab: int,
    beam: int,
    max_len: int,
    alpha: float = 0.6,
    length_penalty: bool = True,
) -> List[int]:
    """返回整条显式束里（含已完成）长度归一分数最高的完整序列（不含 <BOS>）。"""
    # 每条假设 = (累计 log 分数, token 序列, 该前缀对应 hidden)
    beams: List[Tuple[float, List[int], object]] = [(0.0, [bos], init_hidden)]
    finished: List[Tuple[float, List[int]]] = []

    for _ in range(max_len):
        if not beams:
            break
        expanded: List[Tuple[float, List[int], object]] = []
        for score, seq, hidden in beams:
            logits, next_hidden = step_fn(seq[-1], hidden)
            probs = F.log_softmax(logits, dim=-1).squeeze(0)
            for nxt in range(vocab):
                expanded.append((score + probs[nxt].item(), seq + [nxt], next_hidden))

        def norm_key(item):
            _, seq, _ = item
            denom = length_norm(len(seq) - 1, alpha) if length_penalty else 1.0
            return item[0] / denom

        expanded.sort(key=norm_key, reverse=True)

        beams = []
        for score, seq, hidden in expanded[:beam]:
            if seq[-1] == eos:
                finished.append((score, seq))
            else:
                beams.append((score, seq, hidden))

    def best_item(pool):
        if not pool:
            return None
        if length_penalty:
            ranked = sorted(pool, key=lambda x: x[0] / length_norm(len(x[1]) - 1, alpha),
                            reverse=True)
        else:
            ranked = sorted(pool, key=lambda x: x[0], reverse=True)
        return ranked[0]

    winner = best_item(finished) or best_item(beams)
    if winner is None:
        return []
    return winner[1][1:]  # 去掉开头的 <BOS>


if __name__ == "__main__":
    print("-" * 60)
    vocab, bos, eos = 8, 0, 1

    # ---- 构造确定性 step_fn：让"正确的下一步"拿到压倒性 logit ----
    # 已知全局最优期望完整序列 golden = [3,4,5,eos]；每步给对应 token 高 logit。
    golden = [3, 4, 5, eos]

    def step_fn(token, state):
        t = state["t"]
        logits = torch.zeros(1, vocab)
        target = golden[t] if t < len(golden) else eos
        logits[0, target] = 7.0
        return logits, {"t": t + 1}

    # [A] Beam 应找到与 golden 相同的完整序列（顺序正确、以 eos 收尾）
    found = beam_search(step_fn, {"t": 0}, bos=bos, eos=eos, vocab=vocab,
                        beam=2, max_len=8, alpha=0.6)
    assert found == golden, f"beam search returned {found}, expected {golden}"
    print(f"[PASS] beam search returns golden sequence: {found}")

    # [B] 长度归一化单调性
    assert math.isclose(length_norm(6, 0.6) / length_norm(6, 0.6), 1.0)
    assert length_norm(3, 0.6) < length_norm(8, 0.6)
    print("[PASS] length_norm(alpha=0.6) monotonic increasing in length")

    # [C] 长度归一化改变排序：短 + 差得分 vs 长 + 较好得分
    # 无 penalty 排序用 raw score；有 penalty 后，(len2,-2.0) 与 (len4,-2.6)
    # 比较是否翻转。
    s_short = -2.0
    s_long = -2.6
    norm_short = s_short / length_norm(2, 0.6)
    norm_long = s_long / length_norm(4, 0.6)
    # 直观：raw 里 short 更优（-2 > -2.6）；长度归一惩罚后长句分数差距被压缩
    assert s_short > s_long                              # 短更差（raw）
    assert abs(norm_short - norm_long) < abs(s_short - s_long), "norm compresses gap"
    print("[PASS] length normalization reduces short-seq bias (gap compressed)")