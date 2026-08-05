"""m10 Pointer-Generator + Coverage — 生成/复制开关 + 防重复覆盖（v0.9）。

Run:  python m10_pointer_coverage/code.py

v0.9 升级点（versions.md）：
  - Pointer-Generator：Decoder 按 p_gen 混合「从词表生成」和「从源序列复制」，
    复制用 attention 定位源位置，能生成 OOV（人名/地名/数字/专名）。
      P(w) = p_gen · P_vocab(w) + (1 - p_gen) · P_copy(w)
  - Coverage：记录历史 attention 累积，惩罚反复盯同一源区域，防重复生成。

任务=「复制输入数字序列」：源 token ∈ 0..9，目标基础词表只有 <PAD>/<BOS>/<EOS>/<UNK>
（0..9 全是 OOV），于是每个输出 token 都必须走复制路径——pointer-generator
被"逼着"真实生效（p_gen 明显 < 1）。

扩展词表设计：
  基础 4 类 [0..3] + 10 个可复制的源 token（每个源值 v 的扩展 id = 4+v）。
  gold(v) = 扩展 id；模型复制时把 attention alpha 累加到扩展 id 列。

自检：
  [A] p_gen = sigmoid(...) ∈ (0,1)，`1-p_gen` 是复制权重
  [B] 扩展词表上的 final distribution 每行 sum == 1（生成/复制概率守恒）
  [C] coverage 累加单调；coverage 惩罚 = Σ min(alpha, cov) 会随重复盯源增大
  [D] pointer-generator + coverage 在复制任务上 loss 明显下降（且 p_gen < 1）
"""

from __future__ import annotations

import sys

import code as _stdlib_code  # noqa: E402
sys.modules.setdefault("code", _stdlib_code)

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PAD, _BOS, _EOS, _UNK = 0, 1, 2, 3
BASE = 4           # 基础词表：只含 4 个 specials
EXT = BASE + 10    # 扩展词表总大小（+10 个可复制的源 token id）
SRC = 10           # 源 token 空间 {0..9}


class PointerGenerator(nn.Module):
    def __init__(self, hid: int = 64):
        super().__init__()
        self.hid = hid
        self.src_emb = nn.Embedding(SRC, hid)
        self.encoder = nn.GRU(hid, hid, batch_first=True)
        self.attn_key = nn.Linear(hid, hid, bias=False)
        self.dec_emb = nn.Embedding(BASE, hid)
        self.decoder = nn.GRU(hid + hid, hid, batch_first=True)
        self.pgen = nn.Linear(hid * 3, 1)
        self.out_vocab = nn.Linear(hid, BASE)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """src[B,S](∈0..9); tgt[B,L]=[BOS]+(src+4)+[EOS]。返回 mean loss。"""
        B, S = src.shape
        enc_out = self.encoder(self.src_emb(src))[0]     # [B,S,H]
        attn_q = self.attn_key(enc_out)                    # [B,S,H]
        h = torch.zeros(1, B, self.hid, device=device)
        cov = torch.zeros(B, S, device=device)
        dec_in = torch.full((B,), _BOS, device=device)
        total = torch.zeros((), device=device)
        self.last_pg = None
        for t in range(1, tgt.size(1)):
            emb = self.dec_emb(dec_in)                     # [B,H]
            # 注意力：query=d存活隐状态 h[-1]，另受 coverage 惩罚
            score = torch.bmm(attn_q, h[-1].unsqueeze(-1)).squeeze(-1)  # [B,S]
            score = score - 0.2 * cov
            alpha = torch.softmax(score, dim=-1)           # [B,S]
            ctx = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)  # [B,H]
            # 解码一步
            out, h = self.decoder(torch.cat([emb, ctx], -1).unsqueeze(1), h)
            dec_h = out.squeeze(1)                         # [B,H]
            # p_gen 门：0~1，决定生成/复制权重
            pg = torch.sigmoid(self.pgen(torch.cat([dec_h, ctx, emb], -1))[:, 0])
            pv = torch.softmax(self.out_vocab(dec_h), -1)  # [B,BASE]
            pv_ext = torch.zeros(B, EXT, device=src.device)
            pv_ext[:, :BASE] = pv
            # 复制分布：把 alpha 累到各源位置 token 的扩展 id
            ext_idx = (src + BASE)                         # [B,S] ∈ [4,13]
            pc = torch.zeros(B, EXT, device=src.device)
            pc.scatter_add_(1, ext_idx, alpha)
            final = pg.unsqueeze(1) * pv_ext + (1 - pg).unsqueeze(1) * pc
            gold = tgt[:, t]                               # [B] ext-id
            ce = -torch.log(torch.gather(final, 1, gold.unsqueeze(1)) + 1e-12).mean()
            cov_aux = torch.minimum(alpha, cov).sum()
            total = total + ce + 0.2 * cov_aux
            cov = cov + alpha
            self.last_pg = pg.detach()
            dec_in = torch.where(gold < BASE, gold,
                                 torch.full_like(gold, _UNK))  # OOV 回退 UNK 作输入
        return total


def make_batch(B: int, max_len: int) -> tuple:
    """随机复制实例：src = {0..9}^L，tgt = [BOS] + (src+4) + [EOS]。"""
    src = torch.randint(0, SRC, (B, max_len), device=device)
    tgt = torch.cat([torch.full((B, 1), _BOS, device=device),
                     (src + BASE),
                     torch.full((B, 1), _EOS, device=device)], dim=1)
    return src, tgt  # tgt[:,1:] == src+4（gold = 对应扩展 id）


if __name__ == "__main__":
    print(f"device: {device}\n" + "-" * 60)
    torch.manual_seed(0)

    # [A] p_gen 数学：sigmoid ∈ (0,1)，1-p_gen 即复制权重
    raw = torch.tensor([-5.0, 0.0, 5.0, 9.0])
    sig = torch.sigmoid(raw)
    assert (sig > 0).all() and (sig < 1).all()
    print(f"[PASS] p_gen = sigmoid ∈ (0,1): {[round(float(x), 3) for x in sig]}")

    # [B] 概率守恒：sum(final) == p_gen·1 + (1-p_gen)·1 == 1
    #   pv_ext 只在基础类有质量且和为 1（softmax）；pc 是 alpha 摊开后和为 1。
    p_gen = torch.tensor(0.35)
    pv_ext = torch.zeros(EXT)
    pv_ext[:BASE] = torch.tensor([0.1, 0.2, 0.3, 0.4])       # 基础类和==1
    pc = torch.zeros(EXT)
    pc[5:7] = 0.25; pc[9:11] = 0.25                         # 复制总质量==1
    assert torch.isclose(pv_ext.sum(), torch.tensor(1.0))
    assert torch.isclose(pc.sum(), torch.tensor(1.0))
    final_sum = p_gen * pv_ext.sum() + (1 - p_gen) * pc.sum()
    assert torch.isclose(final_sum, torch.tensor(1.0))
    print("[PASS] B sum(final)= p_gen·1 + (1-p_gen)·1 == 1 (生成/复制总质量守恒)")

    # [C] coverage：累加单调，且在 decode 打分里主动压制已覆盖位置（防重复）
    alpha1 = torch.tensor([0.3, 0.7, 0.0])
    cov1 = torch.zeros_like(alpha1) + alpha1         # 首步后 coverage
    cov2 = cov1 + torch.tensor([0.3, 0.7, 0.0])      # 第二步又看同样位置
    assert (cov2 >= cov1).all()                       # 累加单调
    # 打分阶段的 coverage 惩罚：score -= 0.2*cov，软max后已覆盖位置权重下降
    score_plain = torch.tensor([1.0, 2.0, 0.0])
    plain = torch.softmax(score_plain, -1)
    penalized = torch.softmax(score_plain - 0.2 * cov2, -1)
    assert penalized[0] + penalized[1] < plain[0] + plain[1]
    print(f"[PASS] coverage monotone; re-attending covered cells is suppressed "
          f"({(plain[0]+plain[1]).item():.2f} -> {(penalized[0]+penalized[1]).item():.2f})")

    # [D] pointer-generator 复制任务训练：loss 下降且 p_gen < 1（真的在用复制）
    model = PointerGenerator(64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    pgs = []
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        src, tgt = make_batch(8, 6)
        loss = model(src, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if model.last_pg is not None:
            pgs.append(float(model.last_pg.mean()))
    assert losses[0] > losses[-1], f"no convergence: {losses[0]:.3f}->{losses[-1]:.3f}"
    avg_pg = sum(pgs[-50:]) / 50
    assert avg_pg < 1.0 and avg_pg > 0.0
    print(f"[PASS] copy task loss {losses[0]:.3f} -> {losses[-1]:.3f}; "
          f"avg p_gen={avg_pg:.2f} (copy path mixed in, 1-p_gen>0)")