[返回根目录](../README.md)

# m00: Seq2Seq 全景 — 从"固定向量"到"注意力"再到"Transformer"的学习路线


---

## The Problem

Sequence-to-Sequence（Seq2Seq）要解决一个很朴素的问题：**把一段变长的输入序列，映射成另一段长度可能不同的输出序列**——机器翻译（英文→中文）、语音识别（波形→文字）、文本摘要（长文→概要）、对话生成（上文→回复）。

最直接的想法是"先读完整个输入，得到一个能概括它的向量，再用这个向量把输出一个一个吐出来"。这个想法是对的，但立刻会撞上三个由浅到深的困难：

- **难度一（v0.1）**：怎么"读完"输入？RNN 天然是按时间递归的——但"把所有信息压进一个固定向量"之后，输入一长，收尾的信息就把开头的挤掉了。
- **难度二（v0.3）**：生成 "三" 的时候，是不是应该只盯着输入里的 `three`，而不是被迫看一眼整条 `three one nine`？固定 Context 把"该看哪"这件事抹平了。
- **难度三（v1.1）**：RNN 是顺序计算的——第 2 个词必须等第 1 个词算完。能不能让一条序列里的所有位置**同时**彼此交互？

用户需要的不是一句"开箱即用跑个翻译"，而是一个**能把套路一层层拆开、每层都能验证、再逐层拼回完整系统**的过程。

## The Solution

不一次性教完"Attention + Transformer"。**沿着 `versions.md` 的版本树逐级推进**——每一步只新增一个可验证的组件，前面通过验收标准才进入下一步：

```
手写 RNN → 手写 LSTM → 手写 GRU          # 理解"递归"与"门控"
        ↓ 作为 Encoder / Decoder 组件
GRU Encoder-Decoder Seq2Seq (v0.1)       # 最小可运行闭环
        ↓
规范化批处理 (v0.2)                       # 变长 / Padding / Mask
        ↓
Bahdanau Attention (v0.3)                # 动态 Context，突破固定向量瓶颈
        ↓
双向 Encoder (v0.4)                       # 每个位置同时看到左右上下文
        ↓
深层网络 (v0.5)                           # 多层 + Dropout + Residual
        ↓
Beam Search (v0.6)                       # 序列级替 Greedy
        ↓
Scheduled Sampling / Label Smoothing (v0.7)  # 训练-推理分布对齐
        ↓
真实 NLP 数据 (v0.8) + Copy/Coverage (v0.9)  # 子词 / BLEU / OOV / 防重复
        ↓
生产训练 (v1.0)                           # AMP / DDP / Checkpoint
        ↓
Transformer (v1.1)                       # 移除循环依赖，并行训练
```

关键判断：**`versions.md` 定义了一条严格分层的推进顺序**——先保证**数学正确 / 能跑通** → 再**规范化** → 再**提升能力**（Attention/双向/深层）→ 再**改进解码**（Beam/采样）→ 再**转向真实数据** → 最后**生产化**。任何一步都不建议跳级：没跑通 Greedy 解码的翻译，直接上 Beam Search 只会把 Bug 藏进 K 条候选里。

## How It Works

MVP 数据流（一直是"英文数字序列 → 中文数字序列"这条玩具任务，直到 v0.8 才换真实数据）：

```
输入:"three one nine"
     → Encoder（GRU 逐个读入，不断更新隐藏状态）
          → Context = 编码器最终隐藏状态 h_n     # v0.1 固定向量
          ↓
     Decoder 逐词生成：
     <BOS> + h_n → 预测"三"
     "三"  + hidden → 预测"一"
     "一"  + hidden → 预测"九"
                     → 预测 <EOS>，结束
     → "三 一 九"
```

四种"特殊 Token"贯穿全系列：

```
<BOS>：输出序列开始      <EOS>：输出序列结束
<PAD>：批处理补齐        <UNK>：未知词
```

训练用 **Teacher Forcing**（把正确答案喂回 Decoder），推理用 **Greedy / Beam Search**（用自己的预测继续）——两者差异正是 v0.7 要解决的 Exposure Bias。

## Module Map

本系列共 13 个模块（m00–m12），覆盖 `versions.md` 从理论到 v1.1 的完整迭代路线：

| 模块 | 里程碑 | 覆盖版本 | 内容 |
|------|--------|---------|------|
| **m00** | Seq2Seq 全景 | — | 学习路线 + 版本树导览（本文档） |
| **m01** | 手写 RNN / LSTM / GRU Cell | 理论段 | 单隐状态递归 → 门控 Cell State，手动 BPTT |
| **m02** | GRU Encoder-Decoder MVP | v0.1 | 最小可运行闭环，教师强制 + 贪心解码 |
| **m03** | Batch / Padding / Mask / Pack | v0.2 | 变长批处理、`pack_padded_sequence`、Bucketing |
| **m04** | Bahdanau Attention | v0.3 | 动态 Context，突破固定向量瓶颈 |
| **m05** | 双向 Encoder + Bridge | v0.4 | 左右上下文，`[fwd;bwd]` 拼接 |
| **m06** | 多层 + Dropout + Residual + LayerNorm | v0.5 | 深层 Seq2Seq、容量与泛化 |
| **m07** | Beam Search + 长度归一化 | v0.6 | K 候选序列、length penalty |
| **m08** | Scheduled Sampling + Label Smoothing | v0.7 | 缓解 Exposure Bias、降低过度自信 |
| **m09** | Subword + Bucketing + 共享 Embed + BLEU | v0.8 | 子词 / 长度分桶 / Weight Tying / 真实评测 |
| **m10** | Pointer-Generator + Coverage | v0.9 | 复制 OOV、Coverage 防重复 |
| **m11** | AMP / DDP / Checkpoint / 评测体系 | v1.0 | 断点续训、混合精度、多 GPU、日志 |
| **m12** | Transformer Encoder-Decoder | v1.1 | 移除循环依赖，并行训练 |

## 核心推进原则（Key Design Principles）

`versions.md` 的迭代表（第七节）是一套**严格验证驱动**的推进顺序：

> **每个版本都要能"独立验收"**——v0.1 "翻译 loss 下降且贪心解码正确"，v0.2 "padding 不参与 loss/梯度"，v0.3 "Attention 对齐结果与源 token 对应"，v0.6 "Beam Search 优于 Greedy"……

对应到每个 `code.py`，就是它 `__main__` 块里的 `[PASS]` 自检断言：shape 守恒、loss 下降、未来位置概率为 0、解码能复现正确答案。**只有前一个版本的断言通过，才允许进入下一个版本**——这就是本系列"看得清、能验证、可留后退路"的保证。

## Going Deeper

`versions.md` 的理解主线可以压缩成三句话：

```
RNN / LSTM / GRU：  让神经网络拥有"序列状态"，且状态能保存更长期信息
    ↓ 作为编解码组件
Seq2Seq + Attention：用两个网络完成变长序列映射，不再强迫所有输入挤进一个向量
    ↓ 更彻底的改造
Transformer：        不再使用循环，直接让所有位置相互交互、并行训练
```

读完全部 13 个模块，你将亲眼见证"固定 Context → 动态 Attention → 彻底并行"这条 Seq2Seq 的进化主线，并能在手写 `code.py` 中逐层复现。m00 到此给出地图，m01 起开始落地第一块"每块石头"。

## 模块定位

m00 是整个学习路线的**全景导览**：它不写任何代码，只负责在两件事之间建立连接——

1. **学习路线**（本页）：先用《The Problem → The Solution → How It Works》建立系统级心智模型，说清"为什么从手写 RNN Cell 起手"
2. **版本树**（`versions.md`）：把抽象的心智模型对齐到具体的 **v0.1–v1.1** 版本序列与验收标准

读完本模块，你应该已经带着一张清晰地图进入 m01 的第一块 RNN Cell。后续模块的 `[返回根目录](../README.md)` 链接，最终都会回到这张全景图上。
