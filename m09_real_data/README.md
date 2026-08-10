[返回根目录](../README.md)

# m09: 真实 NLP 数据 — Subword + Bucketing + 共享 Embedding + BLEU（v0.8）


---

## The Problem

m02–m07 都跑在 "英文数字↔中文数字" 的 toy 上：空格分词、词表只有 14、不看泛化。`versions.md` 的 v0.8 要求转向**真实 NLP 数据**，这会暴露四个 toy 掩盖不了的问题：

1. **词级词表爆炸 + OOV**：真实语料词表巨大，生词永远出现。→ 需要 **Subword**（BPE）
2. **序列长度分布极不均**：短句 3 个 token、长句 50 个 token，padding 浪费严重。→ **Length Bucketing**
3. **参数冗余**：源与目标共享语言时，各自一份 embedding 太浪费。→ **Shared Embedding (Weight Tying)**
4. **裸 loss 不直观**：翻译质量该用可比的指标。→ **BLEU**

## The Solution

四个机制，各有确定性自检：

```
BPE Subword     →  把一个词拆成可复用子词，遍历词都能 token 化
                                  [A] round-trip 无真 OOV
Length Bucket   →  同 batch 内长度相近（v0.2 已有，此处复用）
Shared Embedding→  encoder/decoder 的 input embedding 与输出投影共享一张表
                                  [B] out_proj.weight IS embed.weight（同一引用）
BLEU            →  n-gram 精确度(clipped) + 简短惩罚
                                  [C] perfect=1.0、disjoint≈0
```

数据自包含：内置一个小英文语料，用 BPE 学到子词词表；任务 = **反转子词序列**
（共享词表 → Weight Tying 成立）。这让"共享 embedding"有了真实数据背景。

## How It Works

### 1. 最小 BPE（字节对编码）

学 merges：统计语料中**相邻符号对**的出现次数，贪心合并最多的那对，反复 N 次：

```
t h e   →  ("t","h") 最常相邻 →  "th" "e"   →  ... → 子词 "the"
```

`_apply_merges(word, merges)` 把词按已学 merge 切回子词；**关键是每轮都从"当前完整 merges 列表"重 token 化**（不是增量 patch），否则丢前序 merge 而让 token 化与词表不一致。这样 round-trip 不产生真 `<UNK>`。

### 2. Length Bucketing

把句子按词数排序后切 batch，同 batch 内长度接近 → padding 列最少（v0.2 的同一手法，这里为真实数据铺路）。

### 3. Shared Embedding / Weight Tying

单一 `embed.weight` 同时供 Encoder 输入、Decoder 输入、以及输出投影复用（转置共享）：

```python
self.out_proj = nn.Linear(hid, vocab_size, bias=False)
self.out_proj.weight = self.embed.weight      # 同一张权重表
```

相比各自独立，省 `(词表大小 × hidden)` 参数，且约束"对应子词的输入与输出表示一致"——正是 m01 用 `nn.Parameter` 埋下的"可共享"伏笔。

### 4. BLEU

```
BLEU = BP · (Σ_n log(clipped n-gram precision)) / N     (N=2)
BP = e^(1 - ref_len/hyp_len)  if hyp 短于 ref
```
n-gram 匹配按**出现次数 clip**（防止重复刷分），再乘 **简短惩罚**（防偏好短句）。

## Code Walkthrough

### Step 1: BPE 学习（每轮重 token）

```python
for _ in range(num_merges):
    pairs = Counter()
    for w, cnt in word_freq.items():
        seq = _apply_merges(w, self.merges)      # 用完整 merges 重切
        for k in range(len(seq) - 1):
            pairs[(seq[k], seq[k+1])] += cnt
    top, n = max(pairs.items(), key=...)          # 最常相邻对
    if n < min_count: break
    self.merges.append(top)
```

### Step 2：Weight Tying（一行绑定）

```python
self.out_proj.weight = self.embed.weight
```

### Step 3：`__main__` 自检

1. `[A]` **BPE round-Trip**：`encode→decode` 还原原文、无 `<UNK>` → `[PASS]`
2. `[B]` **Weight Tying**：`out_proj.weight is embed.weight`（`is` 判定同一对象）→ `[PASS]`
3. `[C]` **BLEU 自洽**：完全一致=1.0、完全无关≈0 → `[PASS]`
4. `[D]` **反转子词任务学习**：loss **5.805→0.0000**（配 Weight-Tying 学会映射）→ `[PASS]`

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **BPE 每轮用完整 merges 重切** | 增量 patch（只重排最新一期）会让 token 化与词表 drift，round-trip 就会吐 `<UNK>`——这是初学者最常踩的 BPE bug，自检 [A] 直接抓它。 |
| **`out_proj.weight = embed.weight`（不是 copy）** | `=` 让两者**共享同一 tensor**，优化时一起更新；若 `copy()` 则各自有参数、断了相关性。[B] 用 `is` 断言对象同一。 |
| **反转任务而非真翻译** | 自包含、无需外部语料即可演示 v0.8 的四套数据机制；且反转天然共享词表，Weight Tying 名正言顺。 |
| **BLEU 在 [C] 只做度量自洽** | BLEU 算法本身（是否 clip、是否加 BP）有没有 bug，用"完美→1、无关→0"钉死，不依赖任何模型。 |

## Going Deeper

- **真实 BPE 会额外做 `</w>` 结尾标记 + 大词表**：这里为教学去掉了词尾切分，所以在 round-trip 里子词能直接拼回词。生产用 SentencePiece / tokenizers 的 BPE。
- **weight tying 限制条件**：只有当源、目标、词表**共用**时成立（如日语子词同表、或同一语言翻译）。中英词表不同就分不开。
- **BLEU 的局限**：只看 n-gram 精确度，不衡量语义；中文/多语更稳的是 chrF/COMET（`versions.md` 的评测表有列）。真正上线看 BLEU，调试看 loss。

## 模块定位

m09 把 toy 升到"真实 NLP 数据"的机制级：子词、分桶、共享 embedding、BLEU——四块全都有确定性自检。m10（v0.9）在 Decoder 上再加 Copy/Coverage，解决生成 OOV 与重复。