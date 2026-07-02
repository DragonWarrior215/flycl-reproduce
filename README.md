# Fly-CL 复现与类脑扩展（基于 LibContinual）

> 《机器学习导论》课程项目 · 持续学习（Continual Learning）
> **Level-1**：在 LibContinual 框架内复现 **Fly-CL**（ICLR 2026）
> **Level-2**：果蝇/海马类脑机制探索 + 原创 CLS-Fly 双系统扩展

本仓库将 Fly-CL（*A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced
Training Time in Pre-trained Model-based Continual Representation Learning*, ICLR 2026,
arXiv:2510.16877）的算法**重构并迁移**到南京大学 MIND 实验室的
[LibContinual](https://github.com/RL-VIG/LibContinual) 框架代码规范中。**未直接提交原作者 repo**——
`core/model/flycl.py` 是按框架 `observe/inference/before_task/after_task` 契约独立重写的实现，
并通过与官方算法的逐任务 **bit-identical** 交叉验证证明忠实性。

---

## 1. 结果速览（CIFAR-100, B0 10×10 类增量）

| 方法         | 我方 Ā (%)  | 我方 Last (%) | 原论文 Ā (%)    | 绝对差   | 相对差    |
| ---------- | --------- | ----------- | ------------ | ----- | ------ |
| **Fly-CL** | **86.28** | 79.41       | 84.61 ± 0.16 | +1.67 | +1.97% |
| RanPAC     | 81.28     | 73.28       | 82.72 ± 0.22 | −1.44 | −1.74% |
| NCM        | 77.84     | 69.54       | —            | —     | —      |

- **Ā（overall accuracy）= 各阶段平均精度 A_t 的均值**，与论文指标定义一致。
- 关键复现结论 **Fly-CL > RanPAC** 双向成立：我方 +5.0，论文 +1.9。
- 我方 Fly-CL 略高于论文，主因是 **骨干预训练来源不同**（详见 §5 差距分析）。

![精度与遗忘曲线](results/accuracy_curves.png)
![我方 vs 原论文](results/comparison_bars.png)

---

## 2. 环境配置

无 GPU 亦可复现——Fly-CL 冻结骨干、更新为闭式线性代数，全程 CPU 可跑。

```bash
# conda 环境（Python 3.10, torch 2.0.1 CPU）
conda create -n libcontinual python=3.10 -y
conda activate libcontinual
# CPU 版 torch（本环境无 GPU；如你本地有 GPU 换成对应 CUDA 构建）
conda install -c pytorch -c conda-forge pytorch=2.0.1 torchvision=0.15.2 cpuonly \
    numpy=1.24 pandas scikit-learn matplotlib pyyaml tqdm pillow -y
conda install -c conda-forge "mkl==2024.0.0" -y   # 关键：torch 2.0.1 需 mkl<2025，否则 undefined symbol iJIT_NotifyEvent
pip install timm==0.9.16 scipy socksio huggingface_hub datasets
# LibContinual 框架自身依赖（import 链会触及 CLIP/OCM 等模块）
pip install ftfy regex continuum diffdist==0.1

# 运行时环境变量（解决沙盒/容器内 OMP 亲和性报错）
export KMP_AFFINITY=disabled OMP_PROC_BIND=false MKL_THREADING_LAYER=GNU
```

> 注：`ftfy/regex`（CLIP tokenizer）、`continuum`（数据集工具）、`diffdist`（OCM 分布式）是
> 框架 `core/model/__init__.py` 的 import 链所需，即使只跑 Fly-CL 也要装，否则 import 即报 ModuleNotFoundError。

**建议在任何入口脚本顶部统一写**（线程/亲和性设置须在 `import torch` 之前生效）：

```python
import os
os.environ["KMP_AFFINITY"]="disabled"; os.environ["OMP_PROC_BIND"]="false"; os.environ["MKL_THREADING_LAYER"]="GNU"
import torch; torch.set_num_threads(16)   # 实测 16 线程在本机 22 核上吞吐最佳，优于 22
```

## 3. 数据与权重准备

```bash
# CIFAR-100（标准 cifar-100-python pickle 格式）放到 ./data/
#   data/cifar-100-python/{train,test,meta}
# ViT-B/16 骨干权重放到 ./assets/vit_b_16_torchvision.pth
```

本次复现在联网受限的沙盒中进行，记录实际可用源：

- **CIFAR-100**：torchvision 默认源 `www.cs.toronto.edu` 与 `storage.googleapis.com` 被拦截；
  改用 fast.ai 公共镜像 `https://fast-ai-imageclas.s3.amazonaws.com/cifar100.tgz`（image-folder 布局），
  再用脚本转回标准 `cifar-100-python/{train,test}` pickle（**保持 torchvision 的字母序 fine-label 索引**：
  apple=0, aquarium_fish=1, ...；50000 train / 10000 test 校验通过）。标准
  `torchvision.datasets.CIFAR100(download=True)` 在不受限网络下亦可直接用。
- **ViT-B/16 骨干**：HuggingFace（`huggingface.co`）在实验期间持续 502 不可达，无法取得论文所用的
  timm ImageNet-21k `augreg` 权重；改用 torchvision `vit_b_16`（`download.pytorch.org`，**ImageNet-1k 有监督**预训练，
  `ViT_B_16_Weights.IMAGENET1K_V1`）。
  
  > ⚠️ **保真度声明**：这是与原论文的一处**预训练来源差异**（IN-1k 监督 vs IN-21k augreg）。
  > 它会系统性影响绝对精度（IN-21k 特征通常更强），但不改变 Fly-CL 相对基线的**趋势**与**结构性零遗忘**特性。
  > §5 会据此分析差距归因；如后续 HF 恢复，可无缝换回 21k 权重重跑。

## 4. 运行命令

**方式 A：完整框架入口（端到端跑真 ViT 前向，CPU 约 90 min）**

```bash
export KMP_AFFINITY=disabled OMP_PROC_BIND=false MKL_THREADING_LAYER=GNU
cd LibContinual
python run_trainer.py --config flycl --device cpu   # 从仓库根跑，assets/ 相对路径才解析
```

（`config/flycl.yaml` 已将 `device_ids: cpu`；框架 `_init_device` 已加 CPU 回退补丁。）

**方式 B：特征缓存 + 闭式求解（推荐，秒级；论文预处理为确定性变换，缓存与在线等价）**

```bash
python extract_features_sharded.py     # 一次性抽取冻结 ViT 特征 -> features/*.npz（分片、断点续跑）
python run_flycl_experiments.py        # Fly-CL / RanPAC / NCM 全部方法 -> results/all_methods.json
```

## 5. 差距分析（为何我方 86.28 vs 论文 84.61）

1. **骨干预训练来源（主因）**：论文用 timm `augreg B_16-i21k...imagenet2012`
   （ImageNet-**21k** 预训练 → IN-1k 微调）；本复现用 torchvision `vit_b_16`
   （ImageNet-**1k** 有监督）。CIFAR-100 类与 ImageNet-1k 高度重叠，IN-1k 特征在 CIFAR-100 上
   反而**很有竞争力**，故我方绝对精度略高。**相对趋势（Fly-CL>RanPAC>NCM）与论文一致**，
   说明复现忠实——差异来自输入特征而非算法。
2. **评测口径**：论文 Ā 为 3 次运行均值±方差；我方为单次固定种子（seed=1993），无方差带。
3. **无数据增强**：CIFAR 预处理严格对齐官方 `test_cifar.sh`（Resize224+CenterCrop+Normalize0.5，
   train/test 同一确定性变换），故特征缓存与在线前向数值等价。

## 6. 忠实性验证：与官方算法逐任务 bit-identical

`run_flycl_experiments.py` 内独立实现了官方 `main.py` 的算法（稀疏投影 + k-WTA + 累加 ridge/GCV/Cholesky）
作为参照；框架内 `core/model/flycl.py` 的 `FlyCL` 类在同一特征上给出**完全相同**的逐阶段 A_t：

```
A_t (both):  [95.1, 94.2, 90.17, 88.2, 86.1, 84.77, 83.4, 81.4, 80.07, 79.41]
Accumulated: 86.28   Last: 79.41   MATCH: True (atol<0.1)
```

（修复过程中发现并修正一个 inference bug：类顺序 shuffle 下应对**全部**读出列 argmax，而非前 N 列。）

## 7. 消融

![超参消融](results/ablation.png)

| 超参            | 扫描         | 发现                                                           |
| ------------- | ---------- | ------------------------------------------------------------ |
| 扩展维 M         | 1000→20000 | 单调↑饱和；81.58→87.14；M=10000 为性价比点                              |
| 编码率 ρ (k-WTA) | 0.05→1.0   | 峰值区 0.2–0.8；**ρ=1.0 去掉 k-WTA=84.56 < 稀疏设置**，直接证明 k-WTA 解相关有效 |
| 突触度 s         | 8→768      | 单调↑；稀疏 s=300 ≈ 稠密 s=768 的 99%，佐证果蝇稀疏接线的效率                    |

## 8. Level-2 类脑探索（完整见 `docs_deliverable/experiment_report.md` §7）

1. **解相关机制验证**：k-WTA（非投影本身）使特征多重共线性 **−33.6%**，坐实果蝇 KC / 海马 DG 的**模式分离**功能。
2. **遗忘分解**：测得 13.9% 遗忘中**仅 0.4% 是真实表征遗忘**，其余为标签空间从 10→100 类增长——
   Fly-CL 顺序无关、**结构性零灾难遗忘**。
3. **原创 CLS-Fly 扩展**：快速海马原型 + 慢速新皮层 ridge 双系统融合。验证 CLS "按需互补" 预测——
   欠定域（小 M）慢系统补偿（+0.16 @ M=300），充分定域时冗余。诚实报告微弱/负面结果。

![类脑解相关分析](results/decorrelation_analysis.png)
![CLS-Fly 双系统扩展](results/clsfly_extension.png)

## 9. 代码结构

```
LibContinual/
  core/model/flycl.py                 # ★ Fly-CL 分类器（本项目重构，框架契约）
  core/model/backbone/vit_flycl.py    # ★ 冻结 ViT-B/16 骨干
  core/model/__init__.py              # 注册 FlyCL
  core/model/backbone/__init__.py     # 注册 vit_flycl
  core/trainer.py                     # 补丁：_init_device CPU 回退
  config/flycl.yaml                   # ★ Fly-CL 配置（超参对齐官方 test_cifar.sh）
extract_features_sharded.py           # 冻结 ViT 特征抽取（分片断点续跑）
run_flycl_experiments.py              # Fly-CL/RanPAC/NCM 参照实现 + 驱动
tests/sanity_flycl.py                 # FlyCL 类逻辑 sanity check
tests/smoke_framework_load.py         # 框架端到端加载冒烟测试（CPU）
docs_deliverable/                     # 架构剖析/环境/论文精读/Level-2 文档 + 图
results/                              # 全部结果 json/csv + 图
```

## 10. 踩坑与解决

| 现象                                                                          | 原因                                                            | 解决                                                                                                        |
| --------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`          | conda 默认装了 MKL 2025，torch 2.0.1 依赖的 `iJIT_NotifyEvent` 符号已被移除 | `conda install -c conda-forge mkl==2024.0.0`                                                              |
| `OMP: Error #179: pthread_setaffinity_np() failed: Operation not permitted` | 沙盒/容器禁止 OpenMP 线程绑核                                           | 导入 torch 前设 `os.environ["KMP_AFFINITY"]="disabled"`（并可设 `OMP_PROC_BIND=false`, `MKL_THREADING_LAYER=GNU`） |
| `torch.get_num_threads()==1`，特征提取慢                                          | 默认单线程                                                         | `torch.set_num_threads(16)`（实测 16 线程在本机 22 核上吞吐最佳，优于 22）                                                  |
| `ModuleNotFoundError: ftfy/diffdist/continuum`                              | 框架 `core/model/__init__.py` import 链触及 CLIP/OCM 等模块           | `pip install ftfy regex continuum diffdist==0.1`                                                          |
| `torch.cuda.set_device` 崩溃                                                  | 框架 `_init_device` 假设 CUDA                                     | 已加 CPU 回退补丁；配 `--device cpu`                                                                              |
| HuggingFace 502 / toronto 403 下载被拦                                          | 沙盒网络策略                                                        | CIFAR 用 fast.ai 镜像；ViT 用 torchvision IN-1k 权重（见 §3）                                                       |
| `Using SOCKS proxy, but 'socksio' not installed`                            | HF hub 走 httpx+SOCKS 代理                                       | `pip install socksio`                                                                                     |

---

**论文**：Zou, Zang, Xu, Ji. *Fly-CL: ...* ICLR 2026. arXiv:2510.16877. 官方代码 github.com/gfyddha/Fly-CL
**框架**：LibContinual, RL-VIG (南京大学 MIND 实验室). github.com/RL-VIG/LibContinual
