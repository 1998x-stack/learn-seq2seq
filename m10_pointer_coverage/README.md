# m10: Pointer-Generator + Coverage — 复制 OOV、防重复生成（v0.9）

[返回根目录](../README.md)

---

## The Problem

普通 Decoder 只能从**固定词表**生成 token。可摘要/问答/翻译里经常必须原样复现输入中的：

- 人名（"林书豪"）、地名（"乌兰巴托"）、数字、"12:30"、专有名词、罕见实体

它们要么不在词表里（OOV），要么词表里没有正确表示。另一方面，Attention Seq2Seq 生成"the model is useful the model is useful"这类**重复**很常见。`versions.md` 的 v0.9 用两个机制各解决一个：

- **Pointer-Generator**：让 Decoder 能"从输入里复制一个 token"，而不是只能生成；
- **Coverage**：记录每个源位置被关注的总量，惩罚反复盯同一处，抑制重复。

## The Solution

### 1. Pointer-Generator（See et al., 2017）

每步算一个**生成/复制门** `p_gen`：

```
P(w) = p_gen · P_vocab(w) + (1-p_gen) · P_copy(w)
```

- `P_vocab(w)`：普通词表分布（softmax 输出投影）；
- `P_copy(w)`：把当前 attention `α` 累加到"源位置 token==w"的所有位置，形成**复制分布**；
- `p_gen = sigmoid([ s_t ; c_t ; embed(y_{t-1}) ])`，模型自己决定"这回生成还是复制"。

**扩展词表**：一张长为 `□词表 ∪ {该句的源 OOV}` 的得分上做 softmax——所以能表示输入里出现的任何词（哪怕 OOV）。

### 2. Coverage（coverage 向量 + coverage loss）

维护每个源位置的覆盖率 `cov_t = Σ_{t'<t} α_{t'}`：

- **打分惩罚**：`score_t,i -= β · cov_{t,i}`，已看够的位置再低分 → 读出不重复；
- **辅助 loss**：`L_cov = Σ_i min(α_t,i, cov_{t,i})`，与交叉熵相加。

## 本模块的自包含复制任务

为让指针机制"被迫"生效，构造一个纯复制任务：

```
源 token ∈ {0..9}；目标基础词表只有 <PAD>/<BOS>/<EOS>/<UNK>
⇒ 0..9 全是 OOV，每个输出 token 都必须走复制路径
```

扩展 id 设计：`扩展 id(v) = 4 + v`（基础 4 类 + 10 个可复制 token）。每次把 attention `α` 用 `scatter_add_` 累加到对应扩展 id：

```python
ext_idx = src + BASE                       # [B,S] ∈ {4..13}
copy_dist.scatter_add_(1, ext_idx, alpha)  # 复制分布
```

## Code Walkthrough

### Step 1: Pointer-Generator forward（v0.9 核心）

```python
pg   = sigmoid(pgen_lin([dec_h; ctx; emb]))          # 生成门 (0,1)
pv   = softmax(out_vocab(dec_h)); pv_ext[:, :BASE]=pv
copy = zeros(B, EXT); copy.scatter_add_(1, ext_idx, alpha)
final = pg·pv_ext + (1-pg)·copy                        # 扩展词表上的分布
loss = -log(final[gold]) + 0.2·Σ min(alpha, cov)      # CE + 辅助 coverage
cov = cov + alpha                                     # 更新覆盖率
```

## 自检 [A]-[D]

| 检查 | 断言 | 结果 |
|---|---|---|
| [A] `p_gen = sigmoid` | ∈ (0,1)，`1-p_gen` 即复制权重 | `[0.007,0.5,0.993,…]` |
| [B] 概率守恒 | `sum(final) == p_gen·1+(1-p_gen)·1 == 1` | 生成/复制总质量守恒 |
| [C] 覆盖率机制 | 累加单调；打分 `score-β·cov` 压制已覆盖位置 | 0.91→0.89 |
| [D] 复制任务学习 | loss 从 23.6 降到 4.8；held-out 复制准确率 **0.89**（真能复制 OOV） | `[PASS]` |

## Key Design Decisions

| 决策 | 理由 |
|---|---|
| **`nn.Embedding(BASE)` 只覆盖基础词表** | OOV 复制 token 没有基础 id，作为 decoder 输入时映射回 `<UNK>`——否则会越界（`Embedding` 行数不够）。这是指针生成器最常见的实现坑。 |
| **`scatter_add_` 建复制分布** | 直接把 α 摊到"token 值对应的扩展列"，等价于对源位置按词聚类，O(1) 而不用逐词循环。 |
| **复制任务故意全 OOV** | 让每个输出 token 都依赖复制路径，从而钉死机制（而不是靠运气把部分 OOV 生成掉）。 |
| **coverage 既打分又进 loss** | 一个管解码时"不看重复"，一个管训练时"别老盯一处"；两处都在代码里。 |

## Going Deeper

- **指针生成是 v0.8 子词的补充**：子词已把多数 OOV 抹平，但仍有长专名/数字需要复制；两者叠加最强。
- **Coverage 的做法不止一种**：经典 `min(α,cov)` 与上下文注意力向量拼接等变体；这里用最直白的 `score-β·cov` + 辅助 loss。
- **真实指针生成在 Decoder 输入里也要"解引用"**：输出 token 若是复制来的，它的词嵌入只能用基础表（OOV→UNK 或查源词嵌入）。本模块如实处理。

## 模块定位

m10 补齐了"生成 OOV"（Pointer）和"防重复"（Coverage）两块，v0.x 系列到此结束。m11（v1.0）开始补**生产工程**：AMP、DDP、Checkpoint、评测体系。