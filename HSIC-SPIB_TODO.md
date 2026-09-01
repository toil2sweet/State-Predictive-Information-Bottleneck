# HSIC-SPIB / HSIC-SPIB+ 实现进度与待办

本文记录相对 `HSIC-SPIB 代码修改参考.md` 与 **HSIC-SPIB+**（向 2024 JCTC MSM 协议靠拢）的落地进度。

---

## HSIC-SPIB（已完成）

1. **`hsic_utils.py`** — RBF / linear / delta kernel；normalized HSIC；梯度只经 \(Z\)。
2. **`SPIB.forward(..., decoder_on_mean=...)`** — HSIC 模式可对 \(\mu\) 解码。
3. **多模式 loss**（`SPIB_training.py`）：`original_spib` / `hsic_spib` / `hybrid_spib`。
4. **State number + TS**：decoder argmax refinement；\(K_i\) margin 过渡态检测与作图。
5. **入口**：`test_model.py` / `test_model_advanced.py` + toy 配置。

推荐 toy 用法：

```bash
python test_model_advanced.py -config examples/Four_Well_hsic_config.ini
python test_model_advanced.py -config examples/Double_Well_hsic_config.ini
```

---

## HSIC-SPIB+（已完成）

目标：在保留 HSIC bottleneck + decoder \(K_i\) 的前提下，并入 2024 式 **过完备动力学初标**、**decoder 头剪枝**、蛋白 **DataNormalize**，使斜排景观 / Trp-cage 上能随 \(\Delta t\) 做 state mining。

### 核心算法

| 改动 | 文件 | 说明 |
|------|------|------|
| Decoder 拆分 + `update_model` | `SPIB.py` | 空/稀有态剪枝时同步缩小 `decoder_output` 与 `output_dim` |
| Refinement 无 KL 也剪枝 | `SPIB_training.py` | 纯 `hsic_spib` 同样 resize decoder |
| `eps_rho` / `encoder_var_mode` | 配置 + 训练 | 人口阈值；`input_dependent` \| `isotropic` |
| `DataNormalize` | `SPIB.py` + `test_model_advanced.py` | `[Data] data_mean` / `data_std` |
| `convergence_history` | 训练 / 保存 | `[refinement, epochs, n_states]` |

### 初标与配置

| 体系 | 准备脚本 | 配置 |
|------|----------|------|
| Müller（斜排） | `muller/prepare_spib_data.py`（`xy_kmeans` K=20） | `examples/Muller_hsic_config.ini` / `examples/Muller_hsic_plus_config.ini` |
| Trp-cage | `trpcage/prepare_spib_data.py`（本地 DESRES DCD → 153 距离 + TICA+kmeans + mean/std） | 2024 原文基线：`examples/TrpCage_sample_plus_config.ini`；HSIC：`examples/TrpCage_hsic_plus_hku_config.ini` |

### 可视化

- `plot_spib_plus.py`：latent 自由能、latent labels、state number vs refinement。
- `test_model_advanced.py` 在 `*plus*` / isotropic / DataNormalize 时自动出 `HSIC_SPIB_plus_*` 图。

### 验收标准

1. **Toy 回归**：`Four_Well_hsic_config` / `Double_Well_hsic_config` 的 state number 与 TS 不退化（已由用户跑通）。
2. **Mülller**：`xy_kmeans` 初标下，learned labels 呈斜排分区；增大 \(\Delta t\) 时 state number 趋向 ~2–3。
3. **Trp-cage**：TICA+kmeans 初标 + normalize + `kernel_x=linear` + `isotropic`；不同 `dt`（如 50 vs 500）下 state number 随分辨率变化；latent FE/labels 可读。本阶段 **不要求** GMRQ/ITS 全套对标，**默认关闭** TS。

### 推荐命令

```bash
# Müller HSIC-SPIB+
python muller/prepare_spib_data.py --method xy_kmeans --n-clusters 20
python test_model_advanced.py -config examples/Muller_hsic_config.ini
python test_model_advanced.py -config examples/Muller_hsic_plus_config.ini

# Trp-cage 2024 SPIB 基线（本地 DESRES DCD）
python trpcage/prepare_spib_data.py
python test_model_advanced.py -config examples/TrpCage_sample_plus_config.ini

# Trp-cage HSIC-SPIB+（同一套 traj_data / 初标）
python test_model_advanced.py -config examples/TrpCage_hsic_plus_hku_config.ini
```

大文件已列入根目录 `.gitignore`（`trpcage/*.npy` 等）。

---

## 尚未实现 / 后续可选

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 中 | CE / HSIC warm-up 多阶段训练 | 文档 Stage 1–3 |
| 中 | Ablation 脚本 | CE-only、HSIC-x-only、HSIC-y-only、hybrid |
| 低 | Trp-cage GMRQ / metastability / ITS | 2024 对照表（本轮明确不做） |
| 低 | 固定 / 可学习 bandwidth 网格搜索 | 现默认 median heuristic |
| 低 | CCA 型 normalized HSIC | 现用 CKA 式 |
| 低 | label imbalance 重加权 | 文档 10.4 |

---

## 设计原则（简记）

- **不换 IB 主目标**：仍用 CE 预测 \(y_{t+\Delta t}\)；HSIC 作 \(I(X,z)\)（及可选 \(I(z,y)\)）代理。
- **保留 decoder**：state refinement 与（可选）TS 依赖 \(K_i\)。
- **向 2024 靠拢的是协议**：过完备初标、剪枝、\(\Delta t\) 控分辨率、蛋白标准化 — 不是改成 VAMP 训练目标。
