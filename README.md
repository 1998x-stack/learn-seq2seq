# Learn Seq2Seq — 从 RNN 到 Transformer 的递进式实现教程（PyTorch）

从零用 PyTorch 搭建 Sequence-to-Sequence 系统，覆盖 `learn-seq2seq/versions.md` 的完整迭代路线：**手写 RNN/LSTM/GRU Cell → GRU Encoder-Decoder MVP（v0.1）→ 规范化批处理（v0.2）→ Bahdanau Attention（v0.3）→ 双向 Encoder（v0.4）→ 深层网络（v0.5）→ Beam Search（v0.6）→ Scheduled Sampling / Label Smoothing（v0.7）→ 真实 NLP 数据（v0.8）→ Copy/Coverage（v0.9）→ 生产训练（v1.0）→ Transformer（v1.1）**。

本教程是 learn-claude-code / learn-llm / learn-deep-research 系列在"从零拆装 Seq2Seq"方向的延伸。每个模块一个目录，含中文教程（Problem → Solution → How It Works → Code Walkthrough → Key Design Decisions → Going Deeper）与自包含可运行 `code.py`。

| 模块 | 里程碑 | 覆盖版本 | 描述 |
|------|--------|---------|------|
| **m00_overview** | Seq2Seq 全景 | — | 学习路线 + 版本树导览 |
| **m01_rnn_cells** | 手写 RNN / LSTM / GRU Cell | 理论段 | 单隐状态递归 → 门控 Cell State |
| **m02_seq2seq_mvp** | GRU Encoder-Decoder MVP | v0.1 | 英文→中文数字序列翻译，Teacher Forcing + Greedy Decoding |
| **m03_batching** | Batch / Padding / Mask / Pack | v0.2 | 变长批处理、`pack_padded_sequence`、Bucketing |
| **m04_attention** | Bahdanau Attention | v0.3 | 动态 Context，突破固定向量瓶颈 |
| **m05_bidirectional** | 双向 Encoder + Bridge | v0.4 | 左右上下文，`[fwd;bwd]` 拼接 |
| **m06_deep** | 多层 + Dropout + Residual + LayerNorm | v0.5 | 深层 Seq2Seq、容量与泛化 |
| **m07_beam_search** | Beam Search + 长度归一化 | v0.6 | K 候选序列、length penalty |
| **m08_scheduled_sampling** | Scheduled Sampling + Label Smoothing | v0.7 | 缓解 Exposure Bias、降低过度自信 |
| **m09_real_data** | Subword + Bucketing + 共享 Embedding + BLEU | v0.8 | 子词 / 长度分桶 / Weight Tying / 真实评测 |
| **m10_pointer_coverage** | Pointer-Generator + Coverage | v0.9 | 复制 OOV、Coverage 防重复 |
| **m11_production** | AMP / DDP / Checkpoint / 评测体系 | v1.0 | 断点续训、混合精度、多 GPU、日志 |
| **m12_transformer** | Transformer Encoder-Decoder | v1.1 | 移除循环依赖，并行训练 |

## Quickstart

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 从 m00 开始阅读
# 每模块目录下均有 README.md（中文教程）+ code.py（可执行代码）

# 3. 运行某个模块（以 m02 为例）
python m02_seq2seq_mvp/code.py
```

## 链接

- [m00_overview/](m00_overview/)
- [m01_rnn_cells/](m01_rnn_cells/)
- [m02_seq2seq_mvp/](m02_seq2seq_mvp/)
- [m03_batching/](m03_batching/)
- [m04_attention/](m04_attention/)
- [m05_bidirectional/](m05_bidirectional/)
- [m06_deep/](m06_deep/)
- [m07_beam_search/](m07_beam_search/)
- [m08_scheduled_sampling/](m08_scheduled_sampling/)
- [m09_real_data/](m09_real_data/)
- [m10_pointer_coverage/](m10_pointer_coverage/)
- [m11_production/](m11_production/)
- [m12_transformer/](m12_transformer/)