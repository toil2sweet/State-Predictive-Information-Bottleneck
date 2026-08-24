# HSIC-SPIB 训练与测试流程概要

本文档分步骤概括 HSIC-SPIB（Hilbert–Schmidt Independence Criterion + State-Predictive Information Bottleneck）的训练与测试过程，并以四势阱（Four-Well）为例说明 **metastable state number** 识别与 **transition state（TS）** 检测。

运行入口（四势阱）：

```bash
python test_model_advanced.py -config examples/Four_Well_hsic_config.ini
```

核心代码：`SPIB.py`（网络）、`SPIB_training.py`（训练/损失/状态数/TS）、`hsic_utils.py`（HSIC）、`plot_transition_states.py`（可视化）。

---

## 0. 方法一句话

HSIC-SPIB 在保留 SPIB 的 **time-lagged 未来态预测 decoder** 的前提下，用 HSIC 作为互信息的可微代理，对潜变量 \(z\) 施加 bottleneck：

- 最大化 \(z\) 与未来态标签 \(y_{t+\Delta t}\) 的依赖；
- 最小化 \(z\) 与当前构象 \(X_t\) 的依赖。

**State number** 与 **TS** 均依赖 decoder 输出的态转移概率 \(K_i(X;\Delta t)\)，不依赖 HSIC 分数本身。

---

## 1. 输入与输出总览

### 1.1 训练输入

| 输入 | 含义 | 四势阱示例 |
|------|------|------------|
| `traj_data` | MD 轨迹坐标/特征，形状 `(N, D)` | `Four_Well_beta3_gamma4_traj_data.npy`，`(300000, 2)`，坐标 \((x,y)\) |
| `initial_labels` | 过完备初始 one-hot 态标签，形状 `(N, C₀)` | `Four_Well_beta3_gamma4_init_label10.npy`，`C₀=10` |
| `dt`（\(\Delta t\)） | 时间滞后（帧数） | 配置中常用 `dt=50`（期望约 4 态） |
| `d` | 潜变量维度（RC 维数） | `d=1` |
| HSIC/训练超参 | `loss_mode`、`beta_x`、`lambda_y`、`eps_ts` 等 | 见 `examples/Four_Well_hsic_config.ini` |

### 1.2 时间滞后样本构造

对轨迹做 time-lagged 配对（`data_init`）：

- 当前输入：\(X_t =\) `past_data`
- 监督目标：\(y_{t+\Delta t} =\) 未来帧对应的 state label
- 丢弃前 `t0` 帧与末尾 `dt` 帧，再 90%/10% 随机划分 train/test

即网络学习：**从当前构象预测 \(\Delta t\) 后进入哪个 metastable state**。

### 1.3 主要输出

训练结束后对整条轨迹保存（`save_traj_results`）：

| 文件 | 内容 |
|------|------|
| `*_data_prediction*.npy` | decoder 概率 \(K_i(X;\Delta t)\)，形状 `(N, C*)` |
| `*_labels*.npy` | hard labels：\(\arg\max_i K_i\) |
| `*_mean_representation*.npy` | latent mean \(\mu(X)\) |
| `*_representation*.npy` | 采样后的 \(z\) |
| `*_state_population*.npy` | 各态人口占比 \(\rho_i\) |
| `*_ts*_mask.npy` 等 | TS 检测结果（若开启） |
| `fig/*.png` | labels、自由能、解析势阱 + TS 叠加图 |

最终 metastable 态数 \(C^*\) 由剪枝后的非空人口决定；四势阱在合适 \(\Delta t\) 下通常 \(C^*=4\)。

---

## 2. 网络结构

模型类：`SPIB`（`SPIB.py`）。

```
X_t  ──► [可选 DataNormalize] ──► Encoder ──► (μ, log σ²)
                                              │
                                    reparameterize → z ~ N(μ, σ²)
                                              │
                         ┌────────────────────┴────────────────────┐
                         │ HSIC 模式常用 decoder_on_mean=True：      │
                         │   用 μ 解码（更稳定）                     │
                         │ 原始 SPIB：用采样 z 解码                  │
                         └────────────────────┬────────────────────┘
                                              ▼
                                    Decoder MLP + LogSoftmax
                                              ▼
                              log K_i(X; Δt)  →  CE 与态标签更新
```

### 2.1 Encoder

- **Linear**（四势阱默认）：\(\mu = W X + b\)；方差网络仍可经隐藏层得到。
- **Nonlinear**：两层 ReLU MLP → \(\mu\)。
- 方差模式：
  - `input_dependent`（默认）：输入相关 \(\log\sigma^2 \in [-10,0]\)
  - `isotropic`：可学习的全局方差（HSIC-SPIB+）

### 2.2 Decoder（必须保留）

- 隐层：\(z \to\) 两层 ReLU MLP（宽度 `neuron_num2`）
- 输出头：`Linear → LogSoftmax`，维数 = 当前态数 \(C\)
- 输出 \(K_i(X;\Delta t)=q_\theta(y_i|z)\)：**态转移概率**

四势阱典型设置：`encoder_type=Linear`，`neuron_num1=neuron_num2=16`，`d=1`。

### 2.3 VampPrior（可选）

原始 SPIB / hybrid 模式用 representative inputs 构造先验 \(r_\theta(z)\) 以计算 KL。纯 `hsic_spib` 模式可不更新 representative，但 decoder 仍完整保留。

---

## 3. 损失函数

三种 `loss_mode`（`SPIB_training.calculate_loss`）：

| 模式 | 公式 | 说明 |
|------|------|------|
| `original_spib` | \(\mathcal{L}=\mathrm{CE}+\beta_{\mathrm{KL}}\,\mathrm{KL}\) | 经典 SPIB |
| `hsic_spib` | \(\mathcal{L}=\mathrm{CE}-\lambda_y\,\widehat{\mathrm{HSIC}}(Z,Y)+\beta_x\,\widehat{\mathrm{HSIC}}(Z,X)\) | **无 KL**；四势阱默认 |
| `hybrid_spib` | \(\mathrm{CE}+\beta_{\mathrm{KL}}\,\mathrm{KL}-\lambda_y\,\mathrm{HSIC}(Z,Y)+\beta_x\,\mathrm{HSIC}(Z,X)\) | 消融对比 |

其中：

- **CE**：对未来 one-hot 标签的交叉熵（加权 NLL），保证 \(K_i\) 可训；
- **HSIC(Z,X)**：压缩项，权重 `beta_x`（最小化对输入细节的依赖）；
- **HSIC(Z,Y)**：预测依赖项，权重 `lambda_y`（最大化与未来态相关）；
- 实际实现中 \(Z\) 取 **\(z_{\mathrm{mean}}=\mu(X)\)** 做 batch HSIC，更稳；
- Kernel：默认 `kernel_z/x=rbf`，`kernel_y=delta`；带宽 median heuristic；`normalized_hsic=True`（CKA 风格归一化）。

四势阱配置：`loss_mode=hsic_spib`，`beta_x=1.0`，`lambda_y=1.0`，`decoder_on_mean=True`。

---

## 4. 训练过程（分步）

### Step 1：读配置与数据

`test_model_advanced.py` 读取 `.ini`，加载轨迹与初始标签，设定 `dt,d,β,lr,seed` 等网格。

### Step 2：构造 time-lagged 数据集

`data_init(t0, dt, …)` → train/test 的 `(past, future, label)`。

### Step 3：初始化 SPIB

- `output_dim = C₀`（四势阱为 10）
- `init_representative_inputs`（若需先验）
- 优化器：Adam；调度：StepLR

### Step 4：小批量训练循环

每个 batch：

1. `forward` → `(log K, z_sample, μ, logσ²)`
2. `calculate_loss` → CE（+ 可选 KL/HSIC）
3. `loss.backward()` + `optimizer.step()`

周期性打印 train/test 的 Loss、CE、KL、HSIC_zx、HSIC_zy，并 checkpoint。

### Step 5：每个 epoch 后的态人口与收敛判据

1. 用当前 decoder 对 **future 构象** 预测概率，取 \(\arg\max\) 得新 hard labels（`update_labels`）
2. 计算态人口 \(\rho_i\)
3. 统计 metastable 态数 \(C=\#\{i:\rho_i>\varepsilon_\rho\}\)
4. 若 \(\|\rho-\rho_{\mathrm{prev}}\|_2 <\) `threshold` 连续超过 `patience` 个 epoch → 认为本轮 refinement 收敛

### Step 6：Label refinement + 态剪枝（识别 state number 的核心）

若 `UpdateLabel=True` 且 refinement 次数 `< refinements`：

1. 用新 labels 替换监督目标
2. `update_model`：丢弃 \(\rho_i\le\varepsilon_\rho\) 的空/稀有态，**缩小 decoder 输出头**，保留存活态权重
3. 重置 optimizer/scheduler，进入下一轮 refinement

重复 Step 4–6，直到达到 `refinements` 次或无法再更新。最终 `output_dim = C*` 即为学到的 **metastable state number**。

### Step 7：训练结束评估

`output_final_result`：在 train/test 上汇总 CE、KL、HSIC 等指标并落盘。

---

## 5. 测试 / 推断过程（分步）

训练完成后（同一脚本内自动进行）：

### Step A：整轨迹前向

对每条轨迹 `save_traj_results`：

- 编码得 \(\mu(X)\)、采样 \(z\)
- 解码得 \(K_i(X;\Delta t)\) 与 hard labels

### Step B：报告 state number

由 `state_population`：\(C^*=\#\{i:\rho_i>0\}\)（或 `>\varepsilon_\rho`）。

### Step C：Transition state 检测（可选）

若 `DetectTransitionStates=True`：

1. 读 `*_data_prediction*.npy` 与人口
2. `identify_transition_states`（见第 7 节）
3. 保存 mask / margin / balance / top2

### Step D：可视化

- 学习态标签图（`plot_state_labels`）
- TS 三联图（`plot_transition_states`）：labels+TS、自由能+TS、解析势+TS
- 可选 HSIC-SPIB+ 潜空间自由能 / 态数随 refinement 曲线（`plot_spib_plus`）

---

## 6. 识别 State Number（以四势阱为例）

### 6.1 问题设定

四势阱解析势沿 \(x\) 有四个阱。轨迹约 \(3\times 10^5\) 帧、2D 坐标。初始标签人为过完备为 **10 类**，真实物理上在合适滞后时间应收敛到 **4 个** metastable states。

配置注释中的经验规律：

| \(\Delta t\) | 期望态数 |
|--------------|----------|
| 50 | ~4 |
| 200 | ~3 |
| 1000 | ~2 |

（更大 \(\Delta t\) 会合并动力学上更快互通的阱。）

### 6.2 识别流程

```
初始 C₀=10 过完备 labels
        │
        ▼
  训练：用 K_i 预测 y_{t+Δt}（+ HSIC bottleneck）
        │
        ▼
  每个 epoch：future 上 argmax(K) → 新 labels → 更新 ρ_i
        │
        ▼
  人口稳定后：剪掉 ρ_i≈0 的态，decoder 头从 10 → … → C*
        │
        ▼
  多次 refinement（四势阱 refinements=8）
        │
        ▼
  最终 C*（典型 dt=50 时为 4）
```

要点：

- **不**预先把态数固定为 4；由数据驱动的人口剪枝得到 \(C^*\)。
- 剪枝与标签更新 **只看 decoder 的 \(K_i\)**，不看 HSIC。
- 收敛条件看 **态人口变化**（`threshold=0.01`，`patience=2`），而非单纯看 loss。

### 6.3 四势阱训练后如何读结果

1. 看日志中 `Metastable state number: …` 与 `After prune: output_dim=…`
2. 看 `*_state_population*.npy` 的非零维数
3. 看 `fig/*_learned_labels_*.png`：应大致对应四个阱区域的分色

---

## 7. 检测 Transition State（以四势阱为例）

### 7.1 判据（decoder \(K_i\) margin）

对每个帧，仅在 **active metastable states** 上取 top-1 / top-2 概率：

\[
\mathrm{Margin}=K_{(1)}-K_{(2)},\qquad
\mathrm{Balance}=1-\mathrm{Margin}
\]

- 候选 TS：\(\mathrm{Margin}<\varepsilon_{\mathrm{ts}}\)
- 两态情形下等价于 \(K_A\approx K_B\approx 0.5\) 附近

四势阱默认：`eps_ts=0.05`。

### 7.2 跨态邻域过滤

若 `ts_require_cross_state=True`（默认）：

- 在时间窗 `ts_window`（默认 1）内，硬标签需覆盖 **至少两个** active states
- 去掉“某态内部偶然概率接近”的假阳性，保留阱间穿越附近的帧

### 7.3 四势阱上的完整检测步骤

1. 训练收敛并得到 \(C^*\)（如 4）与全轨迹 \(K_i(X;\Delta t)\)
2. 由人口确定 active 态索引
3. 计算每帧 Margin / Balance，筛 `Margin < 0.05` 且满足跨态邻域
4. 保存：
   - `*_ts*_mask.npy`：是否为 TS 帧
   - `*_ts*_margin.npy` / `*_ts*_balance.npy`
   - `*_ts*_top2.npy`：每帧 top1/top2 态 id
5. 可视化（`ts_potential=four_well`，`fe_beta=3.0`）：
   - **labels + TS**：在学习态分区图上标 TS 点
   - **free energy + TS**：经验自由能 \(F=-\log P/\beta\) 上标 TS
   - **analytical potential + TS**：在四阱解析势背景上标 TS（应落在阱间垒附近）

物理图像：TS 帧对应“即将从某一阱跳到另一阱”的构象，decoder 对两个（或多个）阱的预测概率接近，故 Margin 小。

---

## 8. 四势阱端到端清单

```text
1. 准备数据
   traj_data: (N,2) 坐标
   init_label10: (N,10) 过完备标签

2. 配置 examples/Four_Well_hsic_config.ini
   dt=50, d=1, Linear encoder, hsic_spib
   UpdateLabel=True, refinements=8
   DetectTransitionStates=True, eps_ts=0.05

3. 运行
   python test_model_advanced.py -config examples/Four_Well_hsic_config.ini

4. 读 state number
   日志 / state_population → C*（期望 ≈4）

5. 读 transition states
   ts_mask 与 fig/*_with_TS.png → 垒区附近的过渡帧
```

---

## 9. 关键设计约束（勿破坏）

1. **必须保留 decoder**：state refinement、\(C^*\)、TS 全部依赖 \(K_i(X;\Delta t)\)。
2. HSIC 只作 bottleneck 正则；**不能**用 HSIC 分数替代 \(K_i\) 做态划分或 TS。
3. Label 更新始终用 decoder \(\arg\max\)，与是否开启 HSIC 无关。
4. 纯 `hsic_spib` 可去掉 KL，但 CE 预测未来态必须保留。

---

## 10. 相关文件索引

| 文件 | 作用 |
|------|------|
| `SPIB.py` | Encoder / Decoder / label 更新 / 态剪枝 |
| `SPIB_training.py` | 数据构造、loss、训练循环、\(C^*\)、TS |
| `hsic_utils.py` | HSIC / kernel / 归一化 |
| `test_model_advanced.py` | 配置驱动训练+测试+绘图入口 |
| `plot_transition_states.py` | TS 与势阱/自由能可视化 |
| `plot_state_labels.py` | 学习态标签图 |
| `examples/Four_Well_hsic_config.ini` | 四势阱 HSIC-SPIB 配置 |
| `HSIC-SPIB 代码修改参考.md` | 设计动机与实现细节长文 |
