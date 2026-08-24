# HSIC-SPIB 代码修改说明：在 SPIB 中引入 HSIC Bottleneck 并保留 state number 与 transition-state identification 功能

## 1. 修改目标概述

本项目目标是在原始 SPIB 代码库的基础上，引入 HSIC（Hilbert-Schmidt Independence Criterion）作为 Information Bottleneck 中 mutual information 的可微 dependence surrogate，从而构造一个 HSIC-regularized SPIB。

原始 SPIB 的目标函数来自 Information Bottleneck：

\[
\mathcal{L}_{IB}=I(z,y)-\beta I(X,z)
\]

其中：

- \(X\)：当前 MD conformation 或其特征；
- \(z\)：encoder 输出的低维 reaction coordinate / latent representation；
- \(y\)：未来 state label，即 \(y_{t+\Delta t}\)；
- \(I(z,y)\)：希望 \(z\) 能预测未来 state；
- \(I(X,z)\)：希望 \(z\) 不保留过多与未来 state prediction 无关的输入细节。

SPIB 原论文中由于 mutual information 难以直接计算，因此使用 variational lower bound，将目标转化为：

\[
\log q_\theta(y_{t+\Delta t}|z_t)
-
\beta
\log
\frac{p_\theta(z_t|X_t)}{r_\theta(z_t)}
\]

其中 decoder \(q_\theta(y|z)\) 输出 state-transition probability，encoder \(p_\theta(z|X)\) 是 Gaussian encoder，\(r_\theta(z)\) 是 VampPrior。

本修改希望引入 HSIC，用 empirical dependence estimation 替代或补充原始 variational bottleneck 项。需要特别注意：本项目不是复现 HSIC Bottleneck 论文中“不使用反向传播、无输出层”的训练方式，而是在 SPIB 现有 PyTorch 训练框架中使用 HSIC loss，并通过标准 autograd 对 encoder、decoder 等所有可训练层一起反向传播优化。

---

## 2. 总体设计原则

### 2.1 必须保留 SPIB 的 decoder / state-transition head

不要删除 SPIB 的 decoder。

原因是 SPIB 的两个核心功能依赖 decoder 输出：

\[
K_i(X;\Delta t,\theta)
=
q_\theta(y_i|z=\mu_\theta(X))
\]

即从当前构象 \(X\) 出发，在 time delay \(\Delta t\) 后进入第 \(i\) 个 state 的预测概率。

SPIB 的 state refinement 使用：

\[
h_i(X)=
\begin{cases}
1, & i=\arg\max_j K_j(X;\Delta t,\theta) \\
0, & \text{otherwise}
\end{cases}
\]

transition-state identification 使用 top-2 probability balance，例如 two-state 情况下：

\[
K_A(X;\Delta t)\approx K_B(X;\Delta t)\approx 0.5
\]

或 multi-state 情况下：

\[
K_{(1)}(X;\Delta t)-K_{(2)}(X;\Delta t)\approx 0
\]

因此，如果完全改成 pure HSIC objective 而不保留 decoder，则只能学习一个 latent representation \(z\)，但无法继续输出 \(K_i(X;\Delta t)\)，也就无法保留 SPIB 的 state number refinement 和 transition-state detection。

---

### 2.2 推荐目标函数：HSIC-regularized SPIB

默认实现建议使用如下 loss：

\[
\mathcal{J}_{\mathrm{HSIC\text{-}SPIB}}
=
\mathrm{CE}
\left(
q_\theta(y_{t+\Delta t}|z_t),
y_{t+\Delta t}
\right)
-
\lambda_y
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
+
\beta_x
\widehat{\mathrm{HSIC}}(Z,X)
\]

其中：

- \(\mathrm{CE}\)：保留原 SPIB 的 future-state prediction cross-entropy；
- \(\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})\)：鼓励 latent representation \(Z\) 与未来 state label 相关；
- \(\widehat{\mathrm{HSIC}}(Z,X)\)：压缩 \(Z\) 对原始输入 \(X\) 的依赖；
- \(\lambda_y\)：future-state dependence 项权重；
- \(\beta_x\)：input compression 项权重。

最小化该目标时：

\[
-\lambda_y \widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
\]

会最大化 \(Z\) 与未来 state label 的依赖；

\[
+\beta_x \widehat{\mathrm{HSIC}}(Z,X)
\]

会最小化 \(Z\) 与原始输入 \(X\) 的依赖。

可选地，保留原 SPIB 的 KL/rate term 作为 ablation：

\[
\mathcal{J}_{\mathrm{hybrid}}
=
\mathrm{CE}
+
\beta_{\mathrm{KL}}
\mathrm{KL}
-
\lambda_y
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
+
\beta_x
\widehat{\mathrm{HSIC}}(Z,X)
\]

其中：

\[
\mathrm{KL}
=
\log p_\theta(z|X)
-
\log r_\theta(z)
\]

默认建议先实现不含 KL 的 HSIC-SPIB，再保留 hybrid 模式用于对比实验。

---

## 3. 需要保留的 SPIB 原有功能

实现时必须保留以下功能不变或尽量兼容：

### 3.1 数据构造

继续使用原 SPIB 的 time-delayed training pairs：

\[
X_t \rightarrow y_{t+\Delta t}
\]

也就是输入当前构象 \(X_t\)，预测未来 label \(y_{t+\Delta t}\)。

保持原始 `dt` / `Delta t` 参数语义不变。

---

### 3.2 Decoder 输出 future-state probability

保留 decoder 的 softmax / log-softmax 输出：

\[
q_\theta(y_{t+\Delta t}|z_t)
\]

该输出后续用于：

1. cross-entropy；
2. label refinement；
3. state population 统计；
4. state number estimation；
5. transition-state detection。

---

### 3.3 Label refinement

保留原 SPIB 的 label update 逻辑：

\[
\hat{y}(X)=\operatorname{onehot}
\left(
\arg\max_i q_\theta(y_i|z=\mu_\theta(X))
\right)
\]

训练过程中仍然允许：

\[
\text{train model}
\rightarrow
\text{update labels}
\rightarrow
\text{retrain / continue training}
\]

最终 state number 仍然由 non-empty state populations 得到：

\[
C_*
=
\sum_i \mathbf{1}[\rho_i>\epsilon_\rho]
\]

其中：

\[
\rho_i
=
\frac{1}{T}
\sum_t
\mathbf{1}[\hat{y}_t=i]
\]

---

### 3.4 Transition-state identification

保留原 SPIB 的 transition-state 后处理方式。

保存每一帧的 state-transition probability：

\[
\mathbf{K}(X_t)
=
[
K_1(X_t;\Delta t),\dots,K_D(X_t;\Delta t)
]
\]

two-state 情况下：

\[
\mathrm{TSScore}(X_t)
=
1-
|K_A(X_t;\Delta t)-K_B(X_t;\Delta t)|
\]

multi-state 情况下：

\[
\mathrm{Margin}(X_t)
=
K_{(1)}(X_t;\Delta t)-K_{(2)}(X_t;\Delta t)
\]

其中 \(K_{(1)}\) 和 \(K_{(2)}\) 是 top-1 和 top-2 predicted probabilities。

低 margin / high balance 的 frames 是 transition-state candidates。建议继续结合 trajectory temporal context 检查该区域前后是否属于不同 metastable states。

---

## 4. HSIC empirical expression

在 mini-batch 内给定：

\[
\mathcal{B}
=
\{(x_i,y_i^+)\}_{i=1}^{B}
\]

其中：

\[
y_i^+=y_{t_i+\Delta t}
\]

encoder 输出：

\[
z_i=f_\theta(x_i)
\]

构造 kernel matrices：

\[
K_X(i,j)=k_X(x_i,x_j)
\]

\[
K_Z(i,j)=k_Z(z_i,z_j)
\]

\[
K_Y(i,j)=k_Y(y_i^+,y_j^+)
\]

中心化矩阵：

\[
H=I-\frac{1}{B}\mathbf{1}\mathbf{1}^{\top}
\]

empirical HSIC：

\[
\widehat{\mathrm{HSIC}}(A,B)
=
\frac{1}{(B-1)^2}
\operatorname{tr}(K_AHK_BH)
\]

因此：

\[
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
=
\frac{1}{(B-1)^2}
\operatorname{tr}(K_ZHK_YH)
\]

\[
\widehat{\mathrm{HSIC}}(Z,X)
=
\frac{1}{(B-1)^2}
\operatorname{tr}(K_ZHK_XH)
\]

建议支持 normalized HSIC / centered kernel alignment 版本，以缓解不同 kernel scale 导致的 loss magnitude 不稳定问题。

---

## 5. 编码器输出的处理方式

原 SPIB encoder 输出：

\[
\mu_\theta(X),\quad \log \sigma_\theta^2(X)
\]

并通过 reparameterization 得到：

\[
z=\mu_\theta(X)+\sigma_\theta(X)\epsilon
\]

本项目建议默认使用 deterministic latent mean 计算 HSIC：

\[
Z=\mu_\theta(X)
\]

原因：

1. HSIC 是 batch-level dependence measure，使用 deterministic \(z=\mu(X)\) 更稳定；
2. SPIB 的 label refinement 和 trajectory saving 本身也主要使用 \(\mu(X)\)；
3. 使用 sampled \(z\) 会引入额外噪声，不利于 HSIC kernel estimation。

实现上建议：

- 原 SPIB encoder 结构可以暂时保留；
- `z_mean` 作为 HSIC 的 \(Z\)；
- decoder 默认也使用 `z_mean` 进行 HSIC-SPIB 模式下的 prediction；
- 如果保留 original / hybrid 模式，可继续支持 `z_sample` 和 KL term。

如果 loss mode 为纯 HSIC-SPIB 且不使用 KL，则 VampPrior、representative inputs、representative weights 可以不参与训练；但为了兼容原始代码，建议不要删除相关函数，而是通过配置开关跳过。

---

## 6. 推荐新增配置参数

在原 SPIB 的 argparse / config 中新增以下参数。

### 6.1 loss mode

新增：

\[
\texttt{loss\_mode}
\]

可选值：

- `original_spib`：保持原始 SPIB loss；
- `hsic_spib`：使用 CE + HSIC bottleneck；
- `hybrid_spib`：使用 CE + original KL + HSIC bottleneck。

默认建议：

\[
\texttt{loss\_mode = hsic\_spib}
\]

---

### 6.2 HSIC 权重

新增：

\[
\lambda_y
\]

对应：

\[
-\lambda_y \widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
\]

新增：

\[
\beta_x
\]

对应：

\[
+\beta_x \widehat{\mathrm{HSIC}}(Z,X)
\]

可选：

\[
\beta_{\mathrm{KL}}
\]

仅在 hybrid 模式下使用。

---

### 6.3 Kernel 类型

建议支持：

\[
k_Z
\]

的类型：

- `rbf`
- `linear`

建议支持：

\[
k_X
\]

的类型：

- `rbf`
- `linear`
- `none`

建议支持：

\[
k_Y
\]

的类型：

- `linear_onehot`
- `delta`

其中 one-hot label kernel 可定义为：

\[
K_Y=YY^\top
\]

或：

\[
K_Y(i,j)=\mathbf{1}[y_i=y_j]
\]

---

### 6.4 Kernel bandwidth

对于 RBF kernel：

\[
k(a_i,a_j)
=
\exp
\left(
-\frac{\|a_i-a_j\|_2^2}{2\sigma^2}
\right)
\]

建议支持：

- fixed bandwidth；
- median heuristic；
- running median heuristic；
- normalized features before kernel computation。

注意：如果使用 median heuristic，建议将 bandwidth estimate 从计算图中 detach，避免 bandwidth 本身引入不稳定梯度。

---

### 6.5 HSIC normalization

建议新增：

\[
\texttt{hsic\_normalized}
\]

用于控制是否使用 normalized HSIC：

\[
\mathrm{nHSIC}(A,B)
=
\frac{
\mathrm{HSIC}(A,B)
}{
\sqrt{
\mathrm{HSIC}(A,A)\mathrm{HSIC}(B,B)
}
+\epsilon
}
\]

默认建议开启 normalized HSIC，因为原始 HSIC 数值容易受 batch size、kernel bandwidth、feature scale 影响。

---

## 7. 代码修改位置说明

以下是建议修改位置。不要重写整个 SPIB 代码库，应尽量复用原 SPIB 结构。

---

### 7.1 新增 HSIC utility 模块

建议新增一个独立文件，例如：

\[
\texttt{hsic\_utils.py}
\]

其中包含以下功能：

1. pairwise distance computation；
2. RBF kernel；
3. linear kernel；
4. label kernel；
5. centering matrix / centered kernel；
6. empirical HSIC；
7. normalized HSIC。

实现时优先参考 HSIC Bottleneck 代码库中的 empirical HSIC 相关函数，但要适配 SPIB 的 PyTorch autograd 训练方式。

注意：

- `K_X` 和 `K_Y` 可以视为常量，不需要梯度；
- `K_Z` 必须保留梯度，使 HSIC loss 能反向传播到 encoder；
- 不要使用 `.detach()` 断开 `z_mean` 到 `K_Z` 的计算图；
- 可以 detach bandwidth；
- 可以 detach input kernel \(K_X\) 和 label kernel \(K_Y\)。

---

### 7.2 修改 SPIB model forward 输出

原 SPIB 中通常已有：

\[
z_{\mathrm{mean}},\quad z_{\mathrm{logvar}},\quad z_{\mathrm{sample}},\quad \log q_\theta(y|z)
\]

需要确认 forward 能返回：

- decoder log probability；
- z_sample；
- z_mean；
- z_logvar。

HSIC-SPIB 中默认使用：

\[
Z=z_{\mathrm{mean}}
\]

作为 HSIC 的 latent representation。

如果原 forward 默认 decoder 输入是 sampled \(z\)，建议在 HSIC-SPIB 模式下支持 decoder 使用 `z_mean`。

---

### 7.3 修改 loss 计算函数

原 SPIB 的 loss 通常包含：

\[
\mathrm{CE}
+
\beta
\left[
\log p_\theta(z|X)-\log r_\theta(z)
\right]
\]

需要扩展为多模式 loss。

#### original_spib

保持原始行为，不改变结果。

\[
\mathcal{J}_{\mathrm{original}}
=
\mathrm{CE}
+
\beta_{\mathrm{KL}}
\mathrm{KL}
\]

#### hsic_spib

使用：

\[
\mathcal{J}_{\mathrm{hsic}}
=
\mathrm{CE}
-
\lambda_y
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
+
\beta_x
\widehat{\mathrm{HSIC}}(Z,X)
\]

#### hybrid_spib

使用：

\[
\mathcal{J}_{\mathrm{hybrid}}
=
\mathrm{CE}
+
\beta_{\mathrm{KL}}
\mathrm{KL}
-
\lambda_y
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
+
\beta_x
\widehat{\mathrm{HSIC}}(Z,X)
\]

loss 函数返回值建议包括：

- total loss；
- CE；
- KL；
- HSIC(Z,Y)；
- HSIC(Z,X)；
- normalized HSIC values；
- current state population；
- optional decoder accuracy。

这些 logging 信息有助于调参和检查 collapse。

---

### 7.4 修改训练循环

训练循环仍使用原 SPIB 的 optimizer、scheduler、epoch、patience、threshold 和 refinement 逻辑。

但需要确保：

1. `data_inputs` 传入 loss 函数，用于计算 \(K_X\)；
2. `data_targets` 传入 loss 函数，用于计算 \(K_Y\)；
3. 每个 mini-batch 内重新计算 \(K_X,K_Y,K_Z\)；
4. HSIC loss 参与标准 backpropagation；
5. encoder 和 decoder 参数都在 optimizer 中；
6. 若 loss mode 不含 KL，则不需要更新 VampPrior / representative inputs；
7. label refinement 时仍使用 decoder probability，而不是 HSIC score。

---

### 7.5 修改 label update 逻辑

原 SPIB 的 `update_labels()` 应保持基本不变：

\[
\hat{y}(X)
=
\operatorname{onehot}
\left(
\arg\max_i q_\theta(y_i|z=\mu_\theta(X))
\right)
\]

注意：

- 不要用 HSIC score 更新 labels；
- HSIC 是 batch-level dependence objective，不提供 per-state probability；
- label update 必须基于 decoder 输出的 state-transition probability；
- 这样才能保留 SPIB 的 state number pruning 功能。

---

### 7.6 修改 trajectory result saving

原 SPIB 保存以下结果：

- sampled representation；
- mean representation；
- data prediction；
- hard labels；
- state population。

需要保持这些输出不变。

可额外保存：

- HSIC latent representation；
- final HSIC(Z,Y)；
- final HSIC(Z,X)；
- loss mode；
- kernel type；
- bandwidth；
- HSIC weights。

但原有文件命名尽量不破坏，以便复用原分析脚本。

---

## 8. State number 功能如何保留

HSIC-SPIB 的 state number 仍通过 SPIB 原逻辑获得：

1. 初始给定 overcomplete labels：

\[
D_{\mathrm{init}}
\]

2. decoder 输出：

\[
K_i(X;\Delta t)
\]

3. label refinement：

\[
\hat{y}(X)=\arg\max_iK_i(X;\Delta t)
\]

4. 统计 state population：

\[
\rho_i=
\frac{1}{T}
\sum_t
\mathbf{1}[\hat{y}_t=i]
\]

5. 以 population threshold 判断 active states：

\[
C_*=
\sum_i \mathbf{1}[\rho_i>\epsilon_\rho]
\]

注意：

- HSIC 不会自动生成新的 output dimension；
- HSIC-SPIB 和原 SPIB 一样，更适合从 overcomplete initial states 中 prune / merge redundant states；
- 如果初始 \(D\) 太小，模型不会自动 grow 出更多 states；
- 因此初始 labels 数量仍建议设置为大于预期 metastable state number。

---

## 9. Transition-state 功能如何保留

HSIC-SPIB 的 transition-state detection 仍基于 decoder prediction：

\[
K_i(X;\Delta t)
\]

而不是 HSIC。

two-state case：

\[
\mathrm{TSScore}(X)
=
1-
|K_A(X;\Delta t)-K_B(X;\Delta t)|
\]

multi-state case：

\[
\mathrm{Margin}(X)
=
K_{(1)}(X;\Delta t)-K_{(2)}(X;\Delta t)
\]

其中 \(K_{(1)}\) 和 \(K_{(2)}\) 是 top-1 和 top-2 state-transition probabilities。

候选 transition states：

\[
\mathrm{Margin}(X)<\epsilon_{\mathrm{TS}}
\]

或选择 lowest-margin / highest-balance percentile。

建议后处理时结合 trajectory temporal context：

- transition segment 前后是否属于不同 states；
- 是否存在 recrossing；
- 是否在 low-population / high-free-energy region；
- 是否与 known collective variables 或 committor analysis 一致。

---

## 10. 训练稳定性注意事项

### 10.1 不要让 HSIC compression 过强

如果 \(\beta_x\) 太大，encoder 可能学到过度压缩的 representation，导致：

- decoder accuracy 降低；
- all states collapse；
- state population 过度集中；
- transition probability 失去区分度。

需要监控：

\[
\mathrm{Var}(Z)
\]

\[
H(\hat{Y})
=
-\sum_i \rho_i\log\rho_i
\]

\[
\mathrm{CE}
\]

\[
\mathrm{HSIC}(Z,Y)
\]

\[
\mathrm{HSIC}(Z,X)
\]

---

### 10.2 HSIC 计算是 \(O(B^2)\)

empirical HSIC 需要构造 \(B\times B\) kernel matrix。

因此原 SPIB 的大 batch size 可能不适合直接使用 HSIC。

建议从：

\[
B=256
\]

或：

\[
B=512
\]

开始调试。

如果内存不足，可进一步考虑：

- smaller batch；
- linear kernel；
- block HSIC；
- random Fourier feature approximation；
- only compute HSIC every few batches。

第一版实现不要求这些高级优化，但需要保留可扩展接口。

---

### 10.3 Kernel bandwidth 很关键

RBF kernel bandwidth 过小会导致 kernel matrix 近似单位阵；过大会导致 kernel matrix 近似常数阵。二者都会让 HSIC 梯度失效。

建议默认使用 median heuristic，并记录每个 batch 的 bandwidth 统计。

对于 \(X\) 高维输入，建议先确保输入已经标准化。

如果原始 \(X\) 是高维坐标或距离矩阵，建议使用：

- normalized structural descriptors；
- PCA/tICA-reduced input；
- 或 linear kernel 作为 baseline。

---

### 10.4 Label imbalance

由于 SPIB 会更新 labels，某些 state 可能变成 empty state，这是原 SPIB 的功能之一。

但 HSIC label kernel 在 label imbalance 时可能不稳定。

建议：

- 保留 CE；
- 使用 class weights 或 balanced sampling 作为可选项；
- 不要过早 label refinement；
- 可以先进行 CE warm-up，再开启 HSIC；
- 可以设置 minimum population threshold，避免早期删除潜在 state。

---

## 11. 建议训练策略

推荐三阶段训练策略：

### Stage 1：Original SPIB warm-up 或 CE warm-up

先训练若干 epochs：

\[
\mathcal{J}=\mathrm{CE}
\]

或：

\[
\mathcal{J}=\mathrm{CE}+\beta_{\mathrm{KL}}\mathrm{KL}
\]

目的：

- 让 decoder 初步学会 future-state prediction；
- 避免 HSIC 在随机 latent 上产生不稳定梯度；
- 稳定 early label refinement。

---

### Stage 2：HSIC-SPIB training

开启：

\[
\mathcal{J}
=
\mathrm{CE}
-
\lambda_y
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
+
\beta_x
\widehat{\mathrm{HSIC}}(Z,X)
\]

可使用 ramp schedule：

\[
\lambda_y(t)
=
\lambda_y^{\max}
\cdot
s(t)
\]

\[
\beta_x(t)
=
\beta_x^{\max}
\cdot
s(t)
\]

其中 \(s(t)\) 从 0 平滑增加到 1。

---

### Stage 3：Label refinement

沿用原 SPIB refinement：

\[
\text{train}
\rightarrow
\text{predict }K_i(X;\Delta t)
\rightarrow
\text{update labels}
\rightarrow
\text{retrain}
\]

每次 label refinement 后，需要重新计算 \(K_Y\) 的 target labels，因为 \(Y_{\Delta t}\) 已经更新。

---

## 12. 推荐实验与 ablation

至少实现以下模式并确保可以通过参数切换：

### Baseline 1：Original SPIB

\[
\mathcal{J}
=
\mathrm{CE}
+
\beta_{\mathrm{KL}}\mathrm{KL}
\]

用于确认原始代码未被破坏。

---

### Baseline 2：CE-only SPIB

\[
\mathcal{J}
=
\mathrm{CE}
\]

用于判断 HSIC 和 KL 的作用。

---

### Variant 1：HSIC compression only

\[
\mathcal{J}
=
\mathrm{CE}
+
\beta_x
\widehat{\mathrm{HSIC}}(Z,X)
\]

用于检验 HSIC 是否可替代 KL/rate compression。

---

### Variant 2：HSIC predictive dependence only

\[
\mathcal{J}
=
\mathrm{CE}
-
\lambda_y
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
\]

用于检验 future-label dependence enhancement。

---

### Variant 3：Full HSIC-SPIB

\[
\mathcal{J}
=
\mathrm{CE}
-
\lambda_y
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
+
\beta_x
\widehat{\mathrm{HSIC}}(Z,X)
\]

---

### Variant 4：Hybrid SPIB

\[
\mathcal{J}
=
\mathrm{CE}
+
\beta_{\mathrm{KL}}\mathrm{KL}
-
\lambda_y
\widehat{\mathrm{HSIC}}(Z,Y_{\Delta t})
+
\beta_x
\widehat{\mathrm{HSIC}}(Z,X)
\]

---

## 13. 需要验证的输出

修改后至少确认以下输出仍然正常：

1. train loss / test loss；
2. CE loss；
3. KL loss，如适用；
4. HSIC(Z,Y)；
5. HSIC(Z,X)；
6. decoder prediction；
7. updated labels；
8. state population；
9. final state number；
10. mean representation；
11. sampled representation，如适用；
12. transition-state score / margin；
13. saved trajectory result files。

---

## 14. 关键实现检查清单

Codex 修改代码时请逐项检查：

- [ ] 原始 `original_spib` 模式结果不应被破坏。
- [ ] 新增 HSIC utility 函数不应 detach `z_mean`。
- [ ] `K_X` 和 `K_Y` 可以 detach。
- [ ] HSIC loss 必须能反传到 encoder。
- [ ] CE loss 必须能反传到 encoder 和 decoder。
- [ ] `update_labels()` 继续使用 decoder prediction。
- [ ] `save_traj_results()` 继续保存原 SPIB 所需输出。
- [ ] 如果不使用 KL，不应强制初始化或更新 VampPrior。
- [ ] 如果使用 hybrid 模式，原 VampPrior 和 representative input 逻辑必须保持可用。
- [ ] 所有新增参数需要写入日志或保存到 config。
- [ ] HSIC batch size 太大时给出 warning。
- [ ] RBF bandwidth 为 0 或 NaN 时需要 fallback。
- [ ] Label collapse 时需要在日志中显示 state population entropy。
- [ ] Transition-state detection 不能直接使用 HSIC score，必须使用 decoder probability balance。

---

## 15. 推荐最终方法命名

建议命名为：

\[
\textbf{HSIC-SPIB}
\]

完整描述：

**HSIC-SPIB: State-Predictive Hilbert-Schmidt Information Bottleneck for Molecular Dynamics**

简短方法描述：

HSIC-SPIB replaces the variational mutual-information bottleneck in SPIB with empirical HSIC dependence regularization while preserving the state-transition decoder. The model is trained by standard back-propagation using a joint objective that encourages the latent representation to be predictive of future metastable states and compressed with respect to the original conformation. Since the decoder is retained, HSIC-SPIB preserves SPIB's iterative label refinement, automatic pruning of redundant states, state-transition density estimation, and committor-like transition-state identification.

---

## 16. 最重要的实现原则

不要把 SPIB 改成 HSIC Bottleneck 论文中的 layer-wise no-backprop model。

本项目需要：

\[
\text{SPIB architecture}
+
\text{SPIB decoder}
+
\text{SPIB label refinement}
+
\text{SPIB trajectory saving}
+
\text{HSIC regularized objective}
+
\text{standard backpropagation}
\]

也就是说：

\[
\boxed{
\text{HSIC 替代或补充的是 bottleneck objective，不替代 SPIB 的 state-transition decoder。}
}
\]

只有这样才能在引入 HSIC 的同时，保留 SPIB 的 state number estimation 和 transition-state identification 功能。