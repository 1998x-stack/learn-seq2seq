# learn-seq2seq — Sequentially Learning Seq2Seq（PyTorch 递进式教程）

Progressive 13-module tutorial (m00–m12) building a Sequence-to-Sequence system from scratch in PyTorch (v0.x → v1.x covers in `versions.md`). Each module is a directory with a Chinese `README.md` tutorial (Problem → Solution → How It Works → Code Walkthrough → Key Design Decisions → Going Deeper) + a self-contained, runnable `code.py`.

This is a **teaching repository**, not an application. Modules are designed to be read in order (m00 → m12). The source-of-truth version tree lives in `versions.md`; each `README.md` table links back to the root via `[返回根目录](../README.md)`. Do not modify unless explicitly asked.

## Architecture

### Module dependency graph (runtime)

Every `code.py` is **self-contained** — later modules re-implement the pieces they need rather than importing earlier modules. The dependency is *conceptual*, not import-based:

```
m01_rnn_cells        RNN / LSTM / GRU Cell（单隐状态 → 门控 Cell State）
        ↓ concept
m02_seq2seq_mvp      GRU Encoder-Decoder MVP (v0.1)
        ↓ concept
m03_batching         Batch / Padding / Mask / Pack (v0.2)
        ↓ concept
m04_attention        Bahdanau Attention (v0.3)
        ↓ concept
m05_bidirectional    双向 Encoder + Bridge (v0.4)
        ↓ concept
m06_deep             多层 + Dropout + Residual + LayerNorm (v0.5)
        ↓ concept
m07_beam_search      Beam Search + 长度归一化 (v0.6)
        ↓ concept
m08_scheduled_sampling  Scheduled Sampling + Label Smoothing (v0.7)
        ↓ concept
m09_real_data        Subword + Bucketing + 共享 Embedding + BLEU (v0.8)
        ↓ concept
m10_pointer_coverage Pointer-Generator + Coverage (v0.9)
        ↓ concept
m11_production       AMP / DDP / Checkpoint / 评测体系 (v1.0)
        ↓ concept
m12_transformer      Transformer Encoder-Decoder (v1.1)
```

Data flows conceptually: `源序列 (x1..xn) → Encoder → Context（或 Attention 加权）→ Decoder → 目标序列 (y1..ym)`.

### Version coverage

Covered versions (per `versions.md` 第七节迭代路线):

| Series | Modules | Milestone |
|---|---|---|
| v0.x (核心迭代) | m02–m10 | 从最小 MVP 到真实数据与 Copy/Coverage |
| v1.0 (生产) | m11 | AMP/DDP/Checkpoint/评测 |
| v1.1 (Transformer) | m12 | Transformer Encoder-Decoder |

m00（全景）与 m01（手写 RNN/LSTM/GRU Cell）是先行基础，不对应具体 v0.x 版本。

## Environment

This teaching series needs **no external API keys**. All modules run locally on CPU with bundled synthetic / small corpora. No `.env` setup required.

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
```

## Commands

```bash
# Setup
pip install -r requirements.txt

# Run a single module
python m02_seq2seq_mvp/code.py
python m04_attention/code.py
python m12_transformer/code.py

# Syntax check only (no heavy compute)
python -c "import ast; ast.parse(open('m02_seq2seq_mvp/code.py').read())"
```

## Key conventions

### README template (learn-claude-code style)

All `README.md` files follow: **Problem → Solution → How It Works → Code Walkthrough → Key Design Decisions → Going Deeper → 模块定位**.

Every module README must:
- Start from `[返回根目录](../README.md)` back-link so learners can return to the module table.
- List the version(s) it covers (matching the root table's `覆盖版本` column and `versions.md`).
- `code.py` is **self-contained and runnable** (`python mXX_*/code.py`) — it never imports from other modules.
- Each module build step carries a **version-tag comment** (e.g. `# v0.1`, `# v0.3`, `# v1.1`) so the incremental narrative stays anchored to `versions.md`.

## Testing

No separate test framework. Each `code.py` **self-verifies in its `__main__` block** (shape assertions, loss-decreases checks, translation smoke test). Syntax can still be checked with `ast.parse`. Do not add pytest to this repo.

## Git

- This is an **independent git repo** initialized at `learn-seq2seq/.git/` — not the parent `Agent-Tutorials` repo.
- `versions.md` is the source-of-truth and is already committed.
- Commit only module/work files inside `learn-seq2seq/`; never stage the parent repo as a nested gitlink.