# Fly-CL 复现与类脑扩展

> 《机器学习导论》课程项目 · 持续学习（Continual Learning）
>
> - Level-1：复现 Fly-CL（ICLR 2026, arXiv:2510.16877），并集成进 LibContinual 框架。
> - Level-2：分析其果蝇/海马类脑机制，做一个 CLS 双系统扩展。

Fly-CL 是一种基于预训练模型的持续表征学习方法：冻结骨干，用一层稀疏随机投影加赢者通吃（k-WTA），
把分类重构为一次闭式的 ridge 回归读出。它无梯度、无重放，跨任务只累加充分统计量，因此在结构上不产生
灾难性遗忘，训练开销也远低于需要迭代优化的同类方法。

本仓库包含两条复现路径：

1. **官方框架复现**（`official/`）：在原作者仓库 [github.com/gfyddha/Fly-CL](https://github.com/gfyddha/Fly-CL)
   上，用其自带脚本在 RTX 4090 上跑 CIFAR-100 / CUB-200-2011 / VTAB 三个数据集，逐数据集对齐论文数值。
2. **LibContinual 集成**（`LibContinual/`）：把算法按 [LibContinual](https://github.com/RL-VIG/LibContinual)
   的 `observe / inference / before_task / after_task` 契约独立重写为 `core/model/flycl.py`，并通过与官方
   算法逐任务数值一致来验证正确性。Level-2 的类脑分析也在这条路径上完成。

---

## 1. 结果速览

### 1.1 官方框架复现（主结果，RTX 4090）

用官方 `scripts/test_{cifar,cub,vtab}.sh` 的原始超参（扩展维 M=10000、突触度 s=300、编码率 ρ=0.3、
ridge 用 GCV 在 1e6–1e10 间自选、seed cifar=1993 / cub=vtab=2023），三个数据集全部复现：

| 数据集 | 论文 Ā (%) | 复现 Ā (%)，IN21k augreg | 差 | 复现 Ā (%)，timm 默认权重 |
| --- | --- | --- | --- | --- |
| CIFAR-100（10 任务） | 93.89 ± 0.12 | **93.88** | −0.01 | 93.20 |
| CUB-200-2011（10 任务） | 93.84 ± 0.18 | **93.84** | ±0.00 | 91.92 |
| VTAB（5 任务，50 类） | 96.54 ± 0.38 | **95.73** | −0.81（约 2σ） | 94.84 |

- Ā（overall / accumulated accuracy）是各阶段平均精度 A_t 的均值，就是官方 `main.py` 打印的 Accumulated Accuracy。
- 论文数值取自 arXiv:2510.16877 Table 1（ViT-B/16）。
- **权重的一个坑**：官方 `load_model.py` 里 `pretrained=True` 在 timm 0.9.16 下实际加载的是
  `augreg2_in21k_ft_in1k`（ImageNet-21k 预训练后又在 IN-1k 上微调），但作者
  `pretrained_model/download.sh` 下载的是纯 IN21k augreg 权重。两种都跑了：用 IN21k-only 权重时三个数据集
  与论文对齐（CUB 连小数点后两位都一致），用 timm 默认权重则在 CUB/VTAB 上低 1.7–1.9 个点。可见论文数值
  对应的是 download.sh 里那份 IN21k 权重。为此在 `load_model.py` 里新增了一个 `vit_base_patch16_224_in21k`
  分支加载它，原有逻辑没动。
- 每任务训练时间（含 GCV 选 λ）：CIFAR ≈ 11.3 s、CUB ≈ 2.6 s、VTAB ≈ 1.4 s；特征提取 ≈ 6.5 / 2.2 / 1.1 s。
  验证了论文关于训练开销低的说法。
- 六份完整逐任务精度矩阵日志在 `results/official_framework/`（两种权重 × 三数据集），汇总见
  `results/official_framework/summary.json`，运行入口 `official/run_all.sh` 与 `official/run_all_in21k.sh`。

### 1.2 LibContinual 集成复现（CIFAR-100, CPU）

这条路径在无 GPU 的沙盒里完成，骨干换成了 torchvision 的 ViT-B/16（ImageNet-1k 有监督预训练，
因为当时 HuggingFace 不可达，取不到论文用的 IN21k augreg 权重）。超参与官方 `test_cifar.sh` 一致。

| 方法 | Ā (%) | Last (%) | 每任务训练时间 |
| --- | --- | --- | --- |
| **Fly-CL** | **86.28** | 79.41 | ~36 s（含 SVD 选 λ） |
| RanPAC | 81.28 | 73.28 | ~26 s |
| NCM | 77.84 | 69.54 | <1 s |

逐阶段 A_t：`[95.1, 94.2, 90.17, 88.2, 86.1, 84.77, 83.4, 81.4, 80.07, 79.41]`。

这里的 86.28 低于论文 Table 1 的 93.89，差距来自骨干：IN-1k 有监督特征弱于 IN21k augreg。1.1 节换回正确权重后
就补齐了。方法层面的相对排序 Fly-CL > RanPAC > NCM 与论文一致。详见 [§9 差距分析](#9-差距分析)。

![精度与遗忘曲线](results/accuracy_curves.png)
![我方 vs 原论文](results/comparison_bars.png)

---

## 2. 方法

冻结 ViT-B/16 输出特征 `x ∈ ℝ^768`。三步，每一步都对应果蝇嗅觉环路的一个环节（PN→KC→APL→MBON）。

![Fly-CL 方法流水线](results/method_pipeline.png)

**(a) 稀疏随机投影（PN→KC）**：投影矩阵 `W ∈ ℝ^{M×768}`，每行随机选 s 列填 `N(0,1)`、其余为 0，
`z = Wx ∈ ℝ^M`（M=10000 ≫ 768，升维）。

**(b) k-WTA 稀疏化（APL 侧抑制）**：保留 `z` 中最大的 `k = ⌈ρM⌉` 个分量，其余置 0，得稀疏码 `Φ(x)`。
这一步模拟 APL 神经元对 KC 的全局抑制，让编码稀疏且去相关。

```
values, idx = topk(z, k);  Φ = zeros_like(z);  Φ[idx] = values
```

**(c) 累加式 ridge 读出（KC→MBON）**：每来一个任务 t 就累加统计量并闭式求解：

```
Q ← Q + Φ(X_t)ᵀ Y_t            # ℝ^{M×C}，互相关
G ← G + Φ(X_t)ᵀ Φ(X_t)         # ℝ^{M×M}，Gram 矩阵
λ ← GCV_select(...)            # 广义交叉验证自动选岭系数
L = cholesky(G + λI); Wo = cholesky_solve(Q, L)
```

推理时 `ŷ = argmax(Φ(x) Wo)`。

**零遗忘从哪来。** `G` 和 `Q` 是对已见全部数据的充分统计量，跨任务只做加法。学完任务 t 时的 `Wo` 精确
等于「把 0..t 所有数据一次性做 ridge 回归」的解，与任务到达顺序无关，也与是否分任务无关。所以旧类的读出
权重不会被新任务覆盖，遗忘在结构上就是 0，这和需要经验回放或蒸馏去对抗遗忘的方法本质不同。

**GCV 选岭。** 对候选 `λ ∈ {10^a}`，先做一次 SVD，再闭式算每个 λ 的广义交叉验证分数
`GCV(λ) = ‖Y−Ŷ‖²/n / (1−df/n)²`，取最小者，避免了对 λ 网格重训。

**复杂度。** 投影加 k-WTA 是 `O(N·s + N·M log k)`，无梯度无反传；ridge 的主成本是 `G`（M×M）的一次
Cholesky，`O(M³)`，`G, Q` 增量累加 `O(N·M)`。整个流程没有 epoch 循环、没有优化器，这是训练时间低的根源。

论文动机是：直接拿预训练特征做相似度匹配会遇到多重共线性（特征维度高度相关、Gram 矩阵病态），
而白化、迭代优化一类的解相关手段对实时低延迟场景太重。Fly-CL 用稀疏投影加 k-WTA 在低复杂度下渐进
消解共线性。这一机制的定量验证见 [§10.1](#101-k-wta-的解相关模式分离)。

---

## 3. 环境配置

### 3.1 官方框架（GPU）

服务器用的是 AutoDL RTX 4090，Python 3.10 / torch 2.1.2+cu118 / timm 0.9.16 / numpy 1.26 / scipy。
官方 readme 建议 torch 1.13.1，但 Fly-CL 全程冻结骨干加闭式解，没遇到任何版本相关问题。

```bash
conda create -n FlyCL python=3.9 && conda activate FlyCL
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
pip install "timm==0.9.16" scipy tqdm "numpy<2.0.0"
```

### 3.2 LibContinual（CPU）

无 GPU 也能完整复现，因为 Fly-CL 是闭式线性代数。

```bash
conda create -n libcontinual python=3.10 -y && conda activate libcontinual
conda install -c pytorch -c conda-forge pytorch=2.0.1 torchvision=0.15.2 cpuonly \
    numpy=1.24 pandas scikit-learn matplotlib pyyaml tqdm pillow -y
conda install -c conda-forge "mkl==2024.0.0" -y   # torch 2.0.1 需 mkl<2025，否则报 undefined symbol iJIT_NotifyEvent
pip install timm==0.9.16 scipy socksio huggingface_hub datasets
pip install ftfy regex continuum diffdist==0.1    # 框架 import 链需要，缺了会 ModuleNotFoundError
```

`ftfy/regex`（CLIP tokenizer）、`continuum`（数据集工具）、`diffdist`（OCM 分布式）是框架
`core/model/__init__.py` 的 import 链所需，即使只跑 Fly-CL 也得装。

在沙盒/容器里，线程亲和性设置要在 `import torch` 之前生效，建议入口脚本顶部统一写：

```python
import os
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["OMP_PROC_BIND"] = "false"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import torch; torch.set_num_threads(16)   # 实测本机 22 核上 16 线程吞吐最好
```

---

## 4. 数据与权重

**数据布局**（官方框架 `--root` 指向的 `data/`）：

```
data/
  cifar-100-python/     # 官方 tar.gz 解包，md5 eb9058c3a382ffc7106e4002c42a8d85
  cub/{train,test}/     # ImageFolder，200 类
  vtab/{train,test}/    # ImageFolder，50 类
```

- **CIFAR-100** 必须用官方原版 tar.gz（本仓库 `data/cifar-100-python-official.tar.gz`）。用别处重建的版本会因
  md5 对不上触发 torchvision 慢速重下载。标准 `torchvision.datasets.CIFAR100(download=True)` 在正常网络下可直接用。
- **ViT-B/16 权重**：论文用 timm `augreg` IN21k。国内网络从镜像下载：
  `HF_ENDPOINT=https://hf-mirror.com hf download timm/vit_base_patch16_224.augreg_in21k`。
  下好后设 `HF_HUB_OFFLINE=1` 离线加载。复现所用权重也存在 `assets/vit_b16_augreg_in21k.safetensors`（本地，未入库）。

LibContinual 那条路径用的是 torchvision ViT-B/16（`ViT_B_16_Weights.IMAGENET1K_V1`），存在
`assets/vit_b_16_torchvision.pth`。这是与论文的一处骨干差异，影响绝对精度但不改变相对趋势，说明见 §9。

---

## 5. 运行

### 5.1 官方框架

```bash
cd official
bash run_all.sh          # timm 默认权重（augreg2_in21k_ft_in1k），三数据集顺序跑
bash run_all_in21k.sh    # download.sh 对应的纯 IN21k 权重，复现论文数值用这个
```

日志输出到 `logs/{cifar100,cub,vtab}[_in21k].log`，本仓库已归档进 `results/official_framework/`。
超参与官方 `scripts/test_*.sh` 完全一致，只把 `--gpu` 改成 0（单卡）。详见 `official/README_REPRO.md`。

### 5.2 LibContinual

方式 A，完整框架入口（端到端跑真 ViT 前向，CPU 约 90 min）：

```bash
export KMP_AFFINITY=disabled OMP_PROC_BIND=false MKL_THREADING_LAYER=GNU
cd LibContinual
python run_trainer.py --config flycl --device cpu   # 从仓库根跑，assets/ 相对路径才解析得到
```

方式 B，特征缓存加闭式求解（推荐，秒级；论文预处理是确定性变换，缓存与在线前向数值等价）：

```bash
python extract_features_sharded.py   # 抽取冻结 ViT 特征 -> features/*.npz（分片、可断点续跑）
python run_flycl_experiments.py      # Fly-CL / RanPAC / NCM -> results/all_methods.json
```

---

## 6. 集成进 LibContinual

### 6.1 框架的模型契约

LibContinual 里每个算法都是一个 `nn.Module`，实现下面几个接口，`core/model/finetune.py` 的 `Finetune`
是最简基类。Trainer 的任务循环大致是：每个任务先 `before_task`，然后按 batch 调 `observe` 并做
`loss.backward() / optimizer.step()`，验证时调 `inference`，任务末调 `after_task`。

| 契约方法 | Fly-CL 的落点 |
| --- | --- |
| `before_task` | 初始化本任务的特征缓冲 |
| `observe(data)` | 冻结前向抽特征、暂存，返回一个 detached 的 0 loss（兼容 Trainer 的 `loss.backward()`） |
| `after_task` | 投影 + k-WTA → 累加 G/Q → GCV 选 λ → Cholesky 求 Wo |
| `inference(data)` | `argmax(Φ Wo)`，全部 100 列，类增量 |

`data` 是 dict，`data['image']`（BCHW）和 `data['label']`（全局标签）。因为 Fly-CL 无梯度，它与 Trainer
默认的反传循环不冲突：真正的「学习」发生在 `observe` 的统计累加和 `after_task` 的闭式求解里，
`observe` 返回 `loss = torch.zeros(1, requires_grad=True)` 只是为了让 `loss.backward()` 不报错。

### 6.2 评估口径

- **A_t**：学完任务 t 后，对已见任务 0..t 的整体准确率。
- **Ā（Accumulated / Average Incremental Acc）**：`mean_t(A_t)`，Fly-CL 论文口径。
- **Last Acc**：学完最后一个任务后的整体准确率 `A_{T-1}`。
- 设置 `task-agnostic` 即类增量（CIL），推理不给任务 ID，Fly-CL 与 RanPAC 都属这类。

---

## 7. 忠实性验证：与官方算法逐任务一致

`run_flycl_experiments.py` 里独立实现了官方 `main.py` 的算法（稀疏投影 + k-WTA + 累加 ridge / GCV / Cholesky）
作参照，框架内 `core/model/flycl.py` 的 `FlyCL` 类在同一批特征上给出完全相同的逐阶段 A_t：

```
A_t (both):  [95.1, 94.2, 90.17, 88.2, 86.1, 84.77, 83.4, 81.4, 80.07, 79.41]
Accumulated: 86.28   Last: 79.41   MATCH: True (atol < 0.1)
```

修复过程中改掉过一个推理 bug：类顺序被 shuffle 时，应对全部读出列取 argmax，而不是前 N 列。

官方框架那条路径（§1.1）则从另一个角度验证了忠实性：换用论文的 IN21k 权重后，三个数据集的 Ā 直接对齐了论文 Table 1。

---

## 8. 消融

![超参消融](results/ablation.png)

| 超参 | 扫描 | 发现 |
| --- | --- | --- |
| 扩展维 M | 1000→20000 | 单调升、饱和：81.58 → 85.14 → 86.28 → 87.14。M 越大解相关空间越充分，但 Cholesky 是 O(M³)，M=10000 是性价比点 |
| 编码率 ρ | 0.05→1.0 | 峰值区宽（0.2–0.8）。ρ=1.0（去掉 k-WTA、完全稠密）反而降到 84.56，直接说明 k-WTA 的稀疏化确实有用；ρ=0.05 过稀疏也差 |
| 突触度 s | 8→768 | 单调升：79.75 → 85.14 → 85.75。稀疏 s=300 已达稠密 s=768 的约 99%，量化了果蝇稀疏接线的效率 |

---

## 9. 差距分析

主结果（§1.1）用论文的 IN21k 权重时已与论文对齐，这里解释 LibContinual 那条路径（§1.2）的 86.28
为什么低于论文 Table 1 的 93.89。

核心是**骨干预训练来源**。论文和官方复现用 ImageNet-21k augreg ViT-B/16；LibContinual 那次因 HuggingFace
不可达，改用 torchvision 的 ImageNet-1k 有监督 ViT-B/16。IN21k augreg 的特征更强、更线性可分，所以绝对精度更高。
官方框架里换权重的对照（IN21k 93.88 vs timm 默认 93.20 vs 更弱的 IN-1k）也印证了这一点：绝对精度主要由骨干
决定，梯度差可达数个点。

由此有个方法论上的观察：在基于预训练模型的持续学习里，骨干选择对绝对精度的影响往往大于持续学习算法本身。
评估一个持续学习算法的贡献，应当控制骨干、看相对增益与遗忘，而不是绝对 Ā。这也是本仓库把「相对排序一致」
（Fly-CL > RanPAC > NCM，在 IN-1k 与 IN21k 下都成立）作为忠实性判据、而非直接横比绝对数字的原因。

---

## 10. Level-2：类脑机制分析与扩展

以 Fly-CL 的果蝇嗅觉环路为起点，结合海马-新皮层互补学习系统（Complementary Learning Systems, CLS）理论，
做两项机制分析和一个扩展。三个尺度上都是「稀疏 + 互补」：果蝇蘑菇体（PN→KC 扩展 + APL 抑制）做气味模式分离，
海马齿状回（内嗅→DG 扩展 + 强抑制）做记忆模式分离，海马-新皮层（快速稀疏编码 + 慢速巩固）避免灾难性遗忘。
Fly-CL 已经实现了前两者（KC/DG 的模式分离），它的累加式 ridge 读出天然顺序无关，对应 CLS 里新皮层慢速稳定
的一面，但缺一条显式的海马快速通路，§10.3 把它补上。

### 10.1 k-WTA 的解相关（模式分离）

问题：Fly-CL 声称稀疏投影加 k-WTA 能消解多重共线性，这真的发生了吗，是投影还是 k-WTA 起作用？

方法：在 CIFAR-100 训练特征的类均衡子集（2000 样本）上，测特征维度间的平均 |非对角相关|（多重共线性的代理指标）。

![类脑解相关分析](results/decorrelation_analysis.png)

| 阶段 | 平均 \|非对角相关\| |
| --- | --- |
| 原始 ViT 特征（768-d） | 0.0713 |
| 稠密随机投影（5000-d） | 0.0822（反而升高） |
| 投影 + k-WTA | 0.0546（比稠密投影低 33.6%） |

结论：解相关来自 k-WTA，不是投影本身。随机稠密投影只是线性混合，不改变甚至略增相关结构；是 k-WTA 的
竞争性稀疏化（赢者通吃即 APL 全局抑制）压下了维度间的冗余共线性。这正对应 KC/DG 的模式分离。k-WTA 后群体
活动率恰为 ρ=0.3，与生物 KC 的低活动率一致。

补充一点，k-WTA 降低的是特征维度的多重共线性（改善 ridge 读出的矩阵条件数，这才是精度提升的来源），
而不是原始余弦几何上的类间可分度——后者在 k-WTA 后反而下降（类内 0.411→0.493，类间 0.147→0.328）。
两者是不同的量。§8 里 ρ=1.0 去掉 k-WTA 使 Ā 从 85.3 降到 84.6，从任务精度侧独立佐证了前者。

### 10.2 遗忘分解

问题：Fly-CL 测得任务0 的「遗忘」是 13.9%，看着不小，这到底是不是灾难性遗忘？

方法：用学完全部 10 个任务后的最终读出权重 Wo，对任务0 的测试样本分别在全部 100 类、以及仅任务0 的 10 类内
评分，与任务0 刚学完时比较。

| 评分方式 | 任务0 精度 |
| --- | --- |
| 刚学完任务0（10 类） | 95.1% |
| 学完 10 任务后，仅在任务0 的 10 类内评分 | 94.7% |
| 学完 10 任务后，在全部 100 类评分（报表「最终」值） | 81.2% |

结论：任务0 的读出权重在学完所有任务后几乎完好（94.7 vs 95.1，只掉 0.4%）。13.9% 的「遗忘」里只有约 0.4%
是真实的表征/权重遗忘，其余约 13.5% 来自标签空间增长（判别问题从 10 类变成 100 类，混淆机会变多）。这与梯度
类方法的灾难性遗忘有本质区别：G、Q 是全部历史数据的充分统计量，跨任务只做加法，顺序无关，旧类权重不被覆盖。
换句话说，遗忘曲线的下降大部分是任务变难，而非知识丢失。

### 10.3 CLS-Fly 扩展

动机：Fly-CL 的 ridge 读出对应「新皮层慢系统」。CLS 理论认为还需要一个海马快系统——快速、稀疏、高可塑的
即时记忆，在数据不足时提供补充信号。

设计：

- 慢系统（新皮层，精确）：Fly-CL 原本的累加式 ridge 读出 Wo，顺序无关、渐近最优。
- 快系统（海马巩固原型）：在 KC 稀疏空间里，每类维护一个巩固的类原型（稀疏码均值）。
- 融合：`logits = (1−β)·标准化(ridge_logits) + β·标准化(prototype_cosine)`。

![CLS-Fly 双系统扩展](results/clsfly_extension.png)

结果：

- 充分定域（M=5000）：β=0（纯 Fly-CL）= 85.14 最优，加入原型只会单调降低精度（β=1.0 纯原型仅 77.58）。
  精确的 ridge 已经最优，粗糙原型只是稀释。
- 欠定域（小 M，ridge 欠定、噪声大）：慢系统开始补偿——

  | M | 纯 Fly-CL（β=0） | 最优融合 | 增益 |
  | --- | --- | --- | --- |
  | 300 | 76.20 | β=0.5 → 76.36 | +0.16 |
  | 500 | 78.97 | β=0.3 → 79.03 | +0.06 |
  | 1000 | 81.58 | β=0.0 → 81.58 | +0.00 |

结论：增益很小（冻结特征下 ridge 本已很强），但方向和对 M 的单调依赖都符合 CLS 的预测：快/精确系统数据不足
（小 M）时，慢/巩固系统提供稳定补充；快系统容量充足时，慢系统变冗余甚至有害。这在一个纯前向、无梯度的分析式
框架里，用可控实验复现了互补学习系统「按需互补」的核心思想。

局限：本扩展停在读出层的原型巩固。更完整的 CLS 还可以引入 (i) 海马式经验回放（在 KC 空间重放稀疏码）、
(ii) 睡眠期离线巩固（周期性把快系统知识蒸馏进 ridge）、(iii) 基于新颖度的可塑性门控（模式分离度决定学习率）。

---

## 11. 代码结构

```
official/
  Fly-CL/                             官方仓库快照（上游 193b1b8，去 .git）
    models/load_model.py                唯一改动：新增 vit_base_patch16_224_in21k 分支
  run_all.sh, run_all_in21k.sh        两种权重的运行入口
  README_REPRO.md                     官方框架复现说明

LibContinual/
  core/model/flycl.py                 Fly-CL 分类器（按框架契约重写）
  core/model/backbone/vit_flycl.py    冻结 ViT-B/16 骨干
  core/model/__init__.py              注册 FlyCL
  core/trainer.py                     补丁：_init_device CPU 回退
  config/flycl.yaml                   Fly-CL 配置（超参对齐官方 test_cifar.sh）
  config/flycl_cub.yaml, flycl_vtab.yaml

extract_features_sharded.py           冻结 ViT 特征抽取（分片、断点续跑）
run_flycl_experiments.py              Fly-CL / RanPAC / NCM 参照实现 + 驱动
tests/                                sanity check 与框架加载冒烟测试
results/                              全部结果 json/csv/png；official_framework/ 为官方框架日志
```

---

## 12. 踩坑记录

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` | conda 默认装了 MKL 2025，torch 2.0.1 依赖的符号被移除 | `conda install -c conda-forge mkl==2024.0.0` |
| `OMP: Error #179: pthread_setaffinity_np() failed` | 沙盒/容器禁止 OpenMP 绑核 | 导入 torch 前设 `KMP_AFFINITY=disabled`（并 `OMP_PROC_BIND=false`, `MKL_THREADING_LAYER=GNU`） |
| `torch.get_num_threads()==1`，特征提取慢 | 默认单线程 | `torch.set_num_threads(16)` |
| `ModuleNotFoundError: ftfy/diffdist/continuum` | 框架 import 链触及 CLIP/OCM 模块 | `pip install ftfy regex continuum diffdist==0.1` |
| `torch.cuda.set_device` 崩溃 | 框架 `_init_device` 假设有 CUDA | 已加 CPU 回退补丁，配 `--device cpu` |
| CIFAR-100 下载龟速 | 本地那份 md5 与官方不符，触发 torchvision 重下载 | 用官方原版 tar.gz（`data/cifar-100-python-official.tar.gz`，md5 eb9058c3…） |
| timm `pretrained=True` 权重与论文对不上 | 0.9.16 默认拉 augreg2_in21k_ft_in1k，非论文的纯 IN21k | 显式加载 `vit_base_patch16_224.augreg_in21k`（见 §1.1） |
| `Using SOCKS proxy, but 'socksio' not installed` | HF hub 走 httpx+SOCKS 代理 | `pip install socksio` |

---

**论文**：Zou, Zang, Xu, Ji. *Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and
Reduced Training Time in Pre-trained Model-based Continual Representation Learning.* ICLR 2026.
arXiv:2510.16877. 官方代码 github.com/gfyddha/Fly-CL

**框架**：LibContinual, RL-VIG（南京大学 MIND 实验室）. github.com/RL-VIG/LibContinual
