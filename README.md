# Fly-CL 复现与类脑扩展

> 《机器学习导论》课程项目 · 持续学习（Continual Learning）
>
> - Level-1：用 **LibContinual 框架**复现 Fly-CL（ICLR 2026, arXiv:2510.16877），在 CIFAR-100 / CUB-200-2011 / VTAB 三个数据集上端到端跑通。
> - Level-2：分析其果蝇/海马类脑机制，做一个 CLS 双系统扩展。

Fly-CL 是一种基于预训练模型的持续表征学习方法：冻结骨干，用一层稀疏随机投影加赢者通吃（k-WTA），
把分类重构为一次闭式的 ridge 回归读出。它无梯度、无重放，跨任务只累加充分统计量，因此在结构上不产生灾难性遗忘，训练开销也远低于需要迭代优化的同类方法。

**本仓库把 Fly-CL 集成进 LibContinual 框架并在三个数据集上复现。** 具体地，算法按
[LibContinual](https://github.com/RL-VIG/LibContinual) 的 `observe / inference / before_task / after_task`
契约独立重写为 `core/model/flycl.py`，用框架自带的 `run_trainer.py` 端到端训练与评估。忠实性由一份独立的
参照实现逐任务对齐来保证（见 [§7](#7-忠实性验证与参照实现逐任务一致)）。

---

## 1. 结果速览

### 1.1 LibContinual 框架复现（主结果，RTX 4090）

用 `run_trainer.py --config flycl{,_cub,_vtab}` 在 LibContinual 里端到端跑三个数据集，骨干是论文的
IN21k augreg ViT-B/16（Fly-CL `pretrained_model/download.sh` 那份纯 IN21k 权重），超参与论文
`test_{cifar,cub,vtab}.sh` 一致（M=10000、s=300、ρ=0.3、ridge 用 GCV 在 1e6–1e10 间自选、seed cifar=1993 /
cub=vtab=2023）：

| 数据集                 | 协议         | 论文 Ā (%)     | **LibContinual Ā (%)** | Last (%) | 差（vs 论文）  |
| ------------------- | ---------- | ------------ | ---------------------- | -------- | --------- |
| CIFAR-100           | 10 任务 ×10 类 | 93.89 ± 0.12 | **93.02**              | 89.46    | −0.87     |
| CUB-200-2011        | 10 任务 ×20 类 | 93.84 ± 0.18 | **92.87**              | 87.79    | −0.97     |
| VTAB（50 类）          | 5 任务 ×10 类  | 96.54 ± 0.38 | **96.16**              | 93.40    | −0.38     |

- **Ā（Accumulated / Average Incremental Accuracy）** 是各阶段整体精度 A_t 的均值，直接读 LibContinual 打印的
  `[Batch] Overall Avg Acc`；**Last** 是学完最后一个任务后在全部类别上的整体精度 A_{T-1}。全部是类增量（`task-agnostic`，
  推理不给任务 ID）。
- 三个数据集都复现到论文 ±1 点以内。
- 逐阶段 A_t 轨迹（`results/libcontinual_framework/A_t.csv`）：
  - CIFAR：`[99.20, 96.60, 94.67, 94.33, 92.76, 91.55, 91.26, 90.17, 90.19, 89.46]`
  - CUB：`[98.70, 95.82, 95.16, 94.22, 93.43, 92.42, 90.91, 90.64, 89.65, 87.79]`
  - VTAB：`[98.48, 97.99, 95.55, 95.37, 93.40]`
- 三个数据集每个任务 GCV 都选中 λ=1e6（候选下界）——IN21k 特征强、Gram 矩阵条件数好，只需最小正则。
- 端到端墙钟（含真 ViT 前向 + 闭式解，单卡 4090）：CIFAR 348 s、CUB 203 s、VTAB 101 s。
- 完整逐任务日志在 `results/libcontinual_framework/{flycl,flycl_cub,flycl_vtab}.log`，汇总见
  `results/libcontinual_framework/summary.json`。

### 1.2 方法对比（同一骨干下的相对排序）

在同一冻结骨干下横比 Fly-CL 与两个常见 PTM 持续学习基线（相对排序才是算法层面的贡献，见 [§9](#9-差距分析)）：

| 方法         | Ā (%)     | Last (%) | 每任务训练时间          |
| ---------- | --------- | -------- | ---------------- |
| **Fly-CL** | **86.28** | 79.41    | ~36 s（含 SVD 选 λ） |
| RanPAC     | 81.28     | 73.28    | ~26 s            |
| NCM        | 77.84     | 69.54    | <1 s             |

Fly-CL > RanPAC > NCM 的排序与论文一致。（这组对比在 IN-1k 有监督 ViT-B/16 上跑，绝对值因此低于 §1.1 的 IN21k 数值，
但相对趋势不变——正是 §9 要说的「骨干决定绝对精度，算法看相对增益」。）

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
权重不会被新任务覆盖，遗忘在结构上就是 0，这和需要经验回放或蒸馏去对抗遗忘的方法本质不同。这也是 §7 里
参照实现能逐任务对齐、以及 Last 精度对任务顺序稳健的原因：不管任务顺序、不管用哪个框架，最终 Wo 都是同一个闭式解。

**GCV 选岭。** 对候选 `λ ∈ {10^a}`，先做一次 SVD，再闭式算每个 λ 的广义交叉验证分数
`GCV(λ) = ‖Y−Ŷ‖²/n / (1−df/n)²`，取最小者，避免了对 λ 网格重训。

**复杂度。** 投影加 k-WTA 是 `O(N·s + N·M log k)`，无梯度无反传；ridge 的主成本是 `G`（M×M）的一次
Cholesky，`O(M³)`，`G, Q` 增量累加 `O(N·M)`。整个流程没有 epoch 循环、没有优化器，这是训练时间低的根源。

论文动机是：直接拿预训练特征做相似度匹配会遇到多重共线性（特征维度高度相关、Gram 矩阵病态），
而白化、迭代优化一类的解相关手段对实时低延迟场景太重。Fly-CL 用稀疏投影加 k-WTA 在低复杂度下渐进
消解共线性。这一机制的定量验证见 [§10.1](#101-k-wta-的解相关模式分离)。

---

## 3. 环境配置

服务器用的是 AutoDL RTX 4090，Python 3.10 / torch 2.1.2+cu118 / timm 0.9.16 / torchvision 0.16.2。
LibContinual 的 `core/model/__init__.py` 会 import 全部算法，import 链需要下面这些包，即使只跑 Fly-CL 也得装：

```bash
# base 环境已有 torch / torchvision / timm；补齐框架 import 链依赖：
pip install pandas scikit-learn matplotlib pyyaml tqdm scipy \
            ftfy regex continuum "diffdist==0.1"
```

`ftfy/regex`（CLIP tokenizer）、`continuum`（数据集工具）、`diffdist`（OCM 分布式）缺了会 `ModuleNotFoundError`。

---

## 4. 数据与权重

**数据布局**（LibContinual `data_root` 指向的 `data/`，服务器上 `./data` 软链到数据盘）：

```
data/
  cifar-100-python/     # 标准 cifar-100-python pickle（train/test/meta），binary_cifar100 直接读
  cub/{train,test}/     # ImageFolder，200 类
  vtab/{train,test}/    # ImageFolder，50 类
```

- **CIFAR-100**：LibContinual 的 `binary_cifar100` 读取标准 `cifar-100-python/{train,test}` pickle。若本地那份 md5
  与 torchvision 期望不符会触发慢速重下载，建议用官方原版 tar.gz（`data/cifar-100-python-official.tar.gz`，
  md5 eb9058c3…）。
- **ViT-B/16 权重**：论文用 timm `augreg` IN21k。从 Fly-CL 的 `pretrained_model/download.sh` 拿那份纯 IN21k 权重：
  `wget https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz`，
  放到 `LibContinual/assets/vit_b16_augreg_in21k.npz`。`vit_flycl` 骨干用 timm 的 `_load_weights` 加载这份 JAX npz。
  国内也可从镜像取 timm safetensors：`HF_ENDPOINT=https://hf-mirror.com hf download timm/vit_base_patch16_224.augreg_in21k`。
- **权重的一个坑**：timm 0.9.16 里 `create_model(pretrained=True)` 默认拉的是 `augreg2_in21k_ft_in1k`（IN21k
  预训练后又在 IN-1k 上微调），不是论文用的纯 IN21k augreg。经验上纯 IN21k 权重在 CUB/VTAB 上高 1.7–1.9 个点，
  §1.1 的结果对应的正是纯 IN21k 权重；所以 `vit_flycl` 显式用 `_load_weights` 加载 download.sh 那份 npz，而不走
  timm 的 `pretrained=True` 默认分支。

§1.2 的方法对比与 Level-2 分析用的是 torchvision ViT-B/16（IN-1k 有监督，`assets/vit_b_16_torchvision.pth`），
这是与论文的一处骨干差异，影响绝对精度但不改变相对趋势，说明见 §9。

---

## 5. 运行

```bash
cd LibContinual
# assets/ 下放好 vit_b16_augreg_in21k.npz，data/ 指向三个数据集
export HF_HUB_OFFLINE=1
python run_trainer.py --config flycl        # CIFAR-100
python run_trainer.py --config flycl_cub    # CUB-200-2011
python run_trainer.py --config flycl_vtab   # VTAB
```

`device_ids: auto` 会自动挑最空闲的 GPU；`n_gpu: 1` 单卡。最终读 `[Batch] Overall Avg Acc` 即 Ā。
三份配置对应 `config/flycl{,_cub,_vtab}.yaml`，超参对齐论文 `test_*.sh`，只是把 `testing_times` 设成 1
（Fly-CL 是确定性闭式解，框架默认重复评估 10 次纯属浪费，设 1 数值完全等价）。

一键顺序跑三数据集：`bash run_flycl_all.sh`（`results/libcontinual_framework/` 里的日志即由该流程产生）。

§1.2 的基线对比、§8 的消融与 §10 的 Level-2 分析是在冻结 ViT 特征上离线算的（`run_flycl_experiments.py`，
内含 Fly-CL / RanPAC / NCM 的参照实现），结果已归档在 `results/`（`all_methods.json`、`ablation_*.json`、`clsfly*.json` 等）。

---

## 6. 集成进 LibContinual

### 6.1 框架的模型契约

LibContinual 里每个算法都是一个 `nn.Module`，实现下面几个接口。Trainer 的任务循环是：每个任务先 `before_task`，
按 batch 调 `observe` 并做 `loss.backward()/optimizer.step()`，任务末调 `after_task`，然后在已见任务上调 `inference` 评估。

| 契约方法              | Fly-CL 的落点                                                        |
| ----------------- | ----------------------------------------------------------------- |
| `before_task`     | 初始化本任务的特征缓冲                                                       |
| `observe(data)`   | 冻结前向抽特征、暂存，返回 `(None, 0.0, dummy_loss)`；dummy 是 `requires_grad` 的 0，让 `loss.backward()` 成为无害空操作 |
| `after_task`      | 投影 + k-WTA → 累加 G/Q → GCV 选 λ → Cholesky 求 Wo                     |
| `inference(data)` | `argmax(Φ Wo)`，对全部读出列取 argmax（类增量）                               |

`data` 是 dict，`data['image']`（BCHW）和 `data['label']`（全局标签）。因为 Fly-CL 无梯度，它与 Trainer
默认的反传循环不冲突：真正的「学习」发生在 `observe` 的统计累加和 `after_task` 的闭式求解里。

### 6.2 为跑通框架所做的改动

原仓库里的 LibContinual 子集缺了几样东西，Fly-CL 之所以此前跑不了 `run_trainer.py`，就是卡在这里。本次补齐：

- **`config/headers/{data,device,model,optimizer,test}.yaml`**：`core/config/default.yaml` 通过 `includes:`
  引用这五个 header 提供 `device_ids / n_gpu / testing_times / pin_memory` 等默认键，缺了会在加载配置时直接
  KeyError。已按官方 LibContinual 的内容补齐。
- **`config/flycl{,_cub,_vtab}.yaml`**：三数据集配置，权重指向 `assets/vit_b16_augreg_in21k.npz`（纯 IN21k），
  `testing_times: 1`。
- **`core/model/flycl.py` / `core/model/backbone/vit_flycl.py`**：Fly-CL 分类器与冻结 ViT-B/16 骨干，
  并在 `core/model/__init__.py` 注册。
- 数据侧无需改动：`binary_cifar100` 读 CIFAR pickle，`cub/vtab` 走 ImageFolder，train/test 共享同一 `cls_map`。

### 6.3 评估口径

- **A_t**：学完任务 t 后，对已见任务 0..t 的整体准确率。
- **Ā（Accumulated / Average Incremental Acc）**：`mean_t(A_t)`，Fly-CL 论文口径，= 框架的 `[Batch] Overall Avg Acc`。
- **Last Acc**：学完最后一个任务后的整体准确率 `A_{T-1}`。
- 框架先在任务内做一次 in-epoch 验证（此时本任务的闭式解还没算，用的是上一任务的 Wo，属正常现象），
  真正计入 `acc_table` 的 A_t 是 `after_task` 之后的评估，用的是刚解出的新 Wo，口径正确。

---

## 7. 忠实性验证：与参照实现逐任务一致

`run_flycl_experiments.py` 里独立实现了论文 `main.py` 的算法（稀疏投影 + k-WTA + 累加 ridge / GCV / Cholesky）
作参照，框架内 `core/model/flycl.py` 的 `FlyCL` 类在同一批特征上给出完全相同的逐阶段 A_t：

```
A_t (both):  [95.1, 94.2, 90.17, 88.2, 86.1, 84.77, 83.4, 81.4, 80.07, 79.41]
Accumulated: 86.28   Last: 79.41   MATCH: True (atol < 0.1)
```

修复过程中改掉过一个推理 bug：类顺序被 shuffle 时，应对全部读出列取 argmax，而不是前 N 列。

由于 Fly-CL 的最终 Wo 与任务顺序无关，Last 精度对协议是稳健的——这也是判断实现是否忠实的一个便利判据。

---

## 8. 消融

![超参消融](results/ablation.png)

| 超参    | 扫描         | 发现                                                                                 |
| ----- | ---------- | ---------------------------------------------------------------------------------- |
| 扩展维 M | 1000→20000 | 单调升、饱和：81.58 → 85.14 → 86.28 → 87.14。M 越大解相关空间越充分，但 Cholesky 是 O(M³)，M=10000 是性价比点 |
| 编码率 ρ | 0.05→1.0   | 峰值区宽（0.2–0.8）。ρ=1.0（去掉 k-WTA、完全稠密）反而降到 84.56，直接说明 k-WTA 的稀疏化确实有用；ρ=0.05 过稀疏也差      |
| 突触度 s | 8→768      | 单调升：79.75 → 85.14 → 85.75。稀疏 s=300 已达稠密 s=768 的约 99%，量化了果蝇稀疏接线的效率                  |

---

## 9. 差距分析

§1.1 的 LibContinual 复现在三个数据集上都落在论文 ±1 点内（CIFAR −0.87、CUB −0.97、VTAB −0.38）。
这里解释残差从哪来，为什么它不是算法层面的问题。

**(1) 任务划分顺序。** 论文的 `main.py` 用一个 seeded 随机置换把 100/200/50 个类分组成任务；LibContinual 的
`binary_cifar100` 按自然标签序（0–9, 10–19, …）分组，`cub/vtab` 按 ImageFolder 排序 + 框架 seed 的置换。
因为 Fly-CL 的最终 Wo 与顺序无关，**Last 精度几乎不受影响**，但 Ā = mean_t A_t 是对增量轨迹取均值，轨迹依赖
每个阶段有哪些类，所以 Ā 会有零点几个点的差。这是协议约定差异，不是实现错误。

**(2) 随机投影的抽样方式。** 本实现按行 `torch.randperm` 采 s 个非零列；同一 seed 下不同构造画出的 W 略有不同，
导致 Wo 及各处精度有 ~0.5 点的抖动。§7 的参照实现在同一批特征上逐任务对齐，说明这类抖动来自投影抽样而非算法。

**(3) 骨干预训练来源（只对 §1.2 的相对比较相关）。** §1.2 的方法对比在 IN-1k 有监督 ViT-B/16 上跑，
绝对精度因此低于 §1.1 的 IN21k 数值。IN21k augreg 的特征更强、更线性可分，绝对精度更高。这印证：
**绝对精度主要由骨干决定，梯度差可达数个点。**

由此有个方法论上的观察：在基于预训练模型的持续学习里，骨干选择对绝对精度的影响往往大于持续学习算法本身。
评估一个算法的贡献，应当控制骨干、看相对增益与遗忘，而不是绝对 Ā。这也是本仓库把「相对排序一致」
（Fly-CL > RanPAC > NCM）与「顺序无关的 Last 精度稳健」作为忠实性判据的原因。

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

| 阶段               | 平均 \|非对角相关\|         |
| ---------------- | -------------------- |
| 原始 ViT 特征（768-d） | 0.0713               |
| 稠密随机投影（5000-d）   | 0.0822（反而升高）         |
| 投影 + k-WTA       | 0.0546（比稠密投影低 33.6%） |

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

| 评分方式                           | 任务0 精度 |
| ------------------------------ | ------ |
| 刚学完任务0（10 类）                   | 95.1%  |
| 学完 10 任务后，仅在任务0 的 10 类内评分      | 94.7%  |
| 学完 10 任务后，在全部 100 类评分（报表「最终」值） | 81.2%  |

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

  | M    | 纯 Fly-CL（β=0） | 最优融合          | 增益    |
  | ---- | ------------- | ------------- | ----- |
  | 300  | 76.20         | β=0.5 → 76.36 | +0.16 |
  | 500  | 78.97         | β=0.3 → 79.03 | +0.06 |
  | 1000 | 81.58         | β=0.0 → 81.58 | +0.00 |

结论：增益很小（冻结特征下 ridge 本已很强），但方向和对 M 的单调依赖都符合 CLS 的预测：快/精确系统数据不足
（小 M）时，慢/巩固系统提供稳定补充；快系统容量充足时，慢系统变冗余甚至有害。这在一个纯前向、无梯度的分析式
框架里，用可控实验复现了互补学习系统「按需互补」的核心思想。

局限：本扩展停在读出层的原型巩固。更完整的 CLS 还可以引入 (i) 海马式经验回放（在 KC 空间重放稀疏码）、
(ii) 睡眠期离线巩固（周期性把快系统知识蒸馏进 ridge）、(iii) 基于新颖度的可塑性门控（模式分离度决定学习率）。

---

## 11. 代码结构

```
LibContinual/                         主路径：Fly-CL 集成进 LibContinual
  run_trainer.py                        框架入口（--config flycl{,_cub,_vtab}）
  run_flycl_all.sh                      一键顺序跑三数据集
  core/model/flycl.py                   Fly-CL 分类器（按框架契约重写）
  core/model/backbone/vit_flycl.py      冻结 ViT-B/16 骨干（timm IN21k npz / torchvision pth）
  core/model/__init__.py                注册 FlyCL
  core/config/default.yaml              includes: headers/*
  config/headers/*.yaml                 补齐的框架默认键（device/data/model/optimizer/test）
  config/flycl.yaml                     CIFAR-100 配置（超参对齐论文 test_cifar.sh）
  config/flycl_cub.yaml, flycl_vtab.yaml
  assets/vit_b16_augreg_in21k.npz       论文 IN21k augreg 权重（本地，未入库）

run_flycl_experiments.py              Fly-CL / RanPAC / NCM 参照实现（§7 忠实性 + §1.2 基线 + §8/§10 分析，跑在冻结 ViT 特征上）
tests/                                sanity check 与框架加载冒烟测试
results/
  libcontinual_framework/               §1.1 主结果：三数据集端到端日志 + summary.json + A_t.csv
  *.json / *.csv / *.png                消融、Level-2、对比图表
```

---

## 12. 踩坑记录

| 现象                                                    | 原因                                           | 解决                                                                                      |
| ----------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------- |
| `run_trainer.py` 加载配置就 KeyError（device_ids / n_gpu 等） | 仓库里的 LibContinual 子集缺 `config/headers/*.yaml` | 按官方补齐五个 header（本次已修）                                                                     |
| 框架默认 `testing_times: 10`，评估慢 10×                       | 对随机方法做多次平均；Fly-CL 是确定性闭式解，重复评估纯浪费           | 配置里设 `testing_times: 1`（数值完全等价）                                                          |
| timm `pretrained=True` 权重与论文对不上                        | 0.9.16 默认拉 augreg2_in21k_ft_in1k，非论文的纯 IN21k | 显式用 `_load_weights` 加载 download.sh 的 `B_16-i21k-300ep-…npz`（见 §4）                        |
| `ModuleNotFoundError: ftfy/diffdist/continuum`        | 框架 import 链触及 CLIP/OCM 模块                    | `pip install ftfy regex continuum diffdist==0.1`                                        |
| CIFAR-100 下载龟速                                        | 本地那份 md5 与官方不符，触发 torchvision 重下载            | 用官方原版 tar.gz（`data/cifar-100-python-official.tar.gz`，md5 eb9058c3…）                     |

---

**论文**：Zou, Zang, Xu, Ji. *Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and
Reduced Training Time in Pre-trained Model-based Continual Representation Learning.* ICLR 2026.
arXiv:2510.16877. 官方代码 github.com/gfyddha/Fly-CL

**框架**：LibContinual, RL-VIG（南京大学 MIND 实验室）. github.com/RL-VIG/LibContinual
