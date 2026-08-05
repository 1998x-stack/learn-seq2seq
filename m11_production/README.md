# m11: 生产训练 — Checkpoint / AMP / DDP / 评测体系（v1.0）

[返回根目录](../README.md)

---

## The Problem

m02–m10 的模型"能跑"，但要**进生产**还差一堆工程能力（`versions.md` 第 1.0 节开列）：

- 训练到一半断了怎么办？→ **Checkpoint / 断点续训**
- GPU 跑得慢/显存不够？→ **混合精度 AMP**
- 一份模型多个卡怎么并行？→ **DDP**
- 谁说了算"模型能不能上线"？→ **评测指标**

v1.0 的目标：把这些生产机制逐个落地，并给每个机制一套**确定性自检**。

## The Solution

用"反转 token 序列"小任务当载体（本环境收敛稳定），叠加四个机制：

| 机制 | 关键 API | 自检 |
|------|----------|------|
| Checkpoint | `state_dict` + `torch.save/load` | 存盘→载入→同输入前向一致 `[A]` |
| AMP | `autocast` + `GradScaler` | loss 下降、无 NaN `[B]` |
| DDP | `DistributedDataParallel` | `.module` 可还原、单步前向跑通 `[C]` |
| 评测 | holdout Exact-Match | 训练后 ≥0.6 `[D]` |

## How It Works

### 1. Checkpoint（断点续训）

把 `model.state_dict()`（+optimizer、step、best_loss）打进字节流存盘；运行时 `load_state_dict` 恢复到同权重的模型。

```python
torch.save({"model": m.state_dict(), "step": step}, buf)
m2 = TinySeq2Seq(); m2.load_state_dict(ckpt["model"])
```

> 判定标准：载入模型对**同一输入**的前向输出与源模型 `allclose`。

### 2. AMP

前向在 `autocast` 下用 fp16 计算，优化器用 `GradScaler` 缩放梯度防下溢：

```python
with torch.autocast(device_type="cuda"):
    out = model(src, tgt); loss = criterion(out, tgt)
scaler.scale(loss).backward()
scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optim); scaler.update()
```

### 3. DDP

`DistributedDataParallel` 把每张卡上的模型副本梯度做 all-reduce。为在单机也能跑通自检，用 `gloo` 后端 `world_size=1`（一个 rank 也合法），`wrapped.module` 仍指回原模型。

### 4. 评测

训练后在 unseen 随机序列上做贪心反转解码，计算 **Exact-Match**（整条预测 == 反转真值）。

## Code Walkthrough

### Step 1: `train_amp`（CUDA 走 autocast+scaler，CPU 纯 fp32）

```python
if scaler is not None:
    with torch.autocast(device_type="cuda"):
        out = model(src, tgt)
        loss = F.cross_entropy(out.reshape(-1, VOCAB), tgt[:, 1:].reshape(-1))
    scaler.scale(loss).backward(); scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim); scaler.update()
```

### Step 2: `__main__` 四段自检

`[A]` 存→载→前向一致 → `[PASS]`；`[B]` AMP loss **1.487→0.339** → `[PASS]`；`[D]` holdout Exact-Match **0.60** → `[PASS]`；`[C]` DDP wrap `.module==model` → `[PASS]`。

## Key Design Decisions

| 决策 | 理由 |
|------|------|
| **用小逆转任务而非大翻译** | 让四种生产机制（而非训练本身）成为关注点，收敛可控、自检稳定。 |
| **DDP 用 world_size=1** | 单机也能跑通"包装→前向→解包"的完整结构；多卡时只需换 `world_size=N` + 每 rank 各自的 device。gloo 后端对 CPU 张量友好。 |
| **AMP 用 `torch.autocast` / `torch.cuda.amp.GradScaler`** | 避免 `torch.amp.GradScaler` 被视为私有导入（pyright）；公开 API 更稳。 |
| **Checkpoint 判据用"前向一致"** | 比"loss 一致"更直接：状态字典载入后模型行为必须逐位一致，否则断点续训不可信。 |

## Going Deeper

- **生产 Checkpoint 不止 state_dict**：还要存 tokenizer/词表版本、优化器与调度器状态、`best_validation_loss`、`global_step`——versions.md 给的字典都在这。
- **AMP 只在 CUDA 有意义**：CPU 上 autocast 是 no-op；`device.type=='cuda'` 才建 scaler。
- **DDP 的跨进程坑**：每 rank 数据要不同（打乱/切分），梯度靠 all-reduce 均一；单进程演示只验证"包装结构正确"。
- **评测要分训练/验证/测试**：本模块演示了机制（Exact-Match 跑得起来），真实上线还要 BLEU/COMET + 三重切分（见 m09 的 BLEU / versions.md）。

## 模块定位

v1.0 把能跑的模型武装到"能生产"：可断点、可提速、可分布、可评测。最后一个版本 v1.1（m12）把整套 RNN 换成 **Transformer**，拆除循环依赖、实现并行训练——这是 seq2seq 的现代化收尾。