# LibContinual 框架剖析（RL-VIG/LibContinual）

> 目的：在动手集成 Fly-CL 前，先厘清框架的核心抽象、训练循环、数据/评估口径与配置规范，
> 为「原始代码 → LibContinual 抽象」的映射（Step 2）打基础。以下内容全部基于对 master 分支源码的阅读。
> 
> 由南京大学 MIND 实验室维护的框架位于 **`github.com/RL-VIG/LibContinual`**（本项目全部基于此仓库）。

## 1. 顶层结构

```
LibContinual/
├── run_trainer.py          # 入口：解析 --config，glob 定位 YAML，构建 Config → Trainer → train_loop()
├── config/                 # YAML 配置（含 backbones/、headers/ 复用片段；zz_* 为各方法配置）
├── core/
│   ├── config/             # Config 类：YAML 解析 + 默认值合并
│   ├── trainer.py          # Trainer：任务循环、优化器/调度器、_train / _validate、指标
│   ├── data/               # ContinualDatasets / SingleDataset：类增量数据切分与 DataLoader
│   ├── model/              # 各算法实现（finetune.py 为基类，lwf/icarl/ranpac/inflora/...）
│   │   ├── backbone/       # resnet / vit / petl(adapter,vpt,ssf) / clip / alexnet
│   │   └── buffer/         # 经验回放缓冲（linearbuffer / herding / online / er）
│   └── utils/
├── reproduce/              # 每个方法一个目录，含复现说明与结果
└── requirements.txt        # torch==2.0.1, torchvision==0.15.2, timm==0.6.7, numpy==1.21.6 ...
```

## 2. 模型契约

每个算法都是一个 `nn.Module`，实现下述接口。`core/model/finetune.py` 的 `Finetune` 是最简参考基类，多数方法直接继承它（如 `LWF(Finetune)`）：

| 方法                                                          | 何时被 Trainer 调用  | 职责                                                           |
| ----------------------------------------------------------- | --------------- | ------------------------------------------------------------ |
| `__init__(backbone, feat_dim, num_class, **kwargs)`         | `_init_model` 时 | 构建分类头、损失、保存超参                                                |
| `before_task(task_idx, buffer, train_loader, test_loaders)` | 每个任务训练前         | 扩展分类头、冻结/复制旧模型、准备任务级状态                                       |
| `observe(data)` → `(pred, acc, loss)`                       | 每个训练 batch      | 前向 + 计算损失（Trainer 负责 `loss.backward()` 与 `optimizer.step()`） |
| `inference(data[, task_id])` → `(pred, acc)`                | 验证时             | 前向预测；task_id>-1 为 task-aware，None 为 task-agnostic（类增量）       |
| `after_task(task_idx, buffer, train_loader, test_loaders)`  | 每个任务训练后         | 更新原型/缓冲、蒸馏教师快照、偏置校正等                                         |
| `get_parameters(config)`                                    | `_init_optim` 时 | 返回交给优化器的参数组（可只返回可训练子集）                                       |

`data` 是一个 dict：`data['image']`（BCHW 张量）、`data['label']`（类别标签，全局 0..C-1）。

### 关键点：Fly-CL 与该契约的契合

Fly-CL 是**解析式**方法（冻结骨干 + 稀疏随机投影 + 闭式 ridge 回归），没有梯度训练。
因此它天然映射为：

- `observe()` 不做反传，而是把当前 batch 的（投影+WTA）特征累加进协方差统计 `G` 与互相关 `Q`；
- `after_task()` 求解 `Wo=(G+λI)⁻¹Q`（Cholesky），更新读出权重；
- `inference()` 用 `Φ(x)·Wo` 取 argmax。
  
  > 因为 `G,Q` 跨任务累加，闭式解等价于对「已见全部数据」做 ridge 回归 —— **结构上零遗忘**。

## 3. 训练循环（core/trainer.py `train_loop`）

```
for task_idx in range(task_num):
    model.before_task(task_idx, buffer, train_loader[task_idx], test_loader[task_idx])
    _init_optim()                          # 每任务重建优化器/调度器
    for epoch in range(init_epoch if task0 else epoch):
        _train(epoch, dataloader)          # 内部对每 batch 调 model.observe(batch)
        if epoch % val_per_epoch == 0:
            _validate(task_idx)            # 见 §4
    model.after_task(task_idx, buffer, ...)
    # BiC 等有 stage2_train（偏置校正）分支
```

- `_train`：`output, acc, loss = model.observe(batch)`；`loss.backward(); optimizer.step()`。
- `init_epoch` 用于第 0 个任务，`epoch` 用于后续任务（PTM 方法常设 `epoch:1`）。

## 4. 评估与指标（core/trainer.py `_validate`）

- `testing_per_task=True`：逐旧任务子集测试（task-aware 用 `task_id=t`；task-agnostic 用全局 argmax），
  返回 `avg_acc`（全体样本准确率）与 `per_task_acc`（每任务准确率列表）。
- `testing_per_task=False`：把 `<=task_idx` 的测试集合并后统一推理，按类边界分桶统计每任务准确率。
- `setting: task-agnostic` = 类增量（CIL），推理时不给任务 ID —— Fly-CL / RanPAC 都属此类。

**报告口径**（Step 6 对齐用）：

- **每步平均增量精度** `A_t` = 学完任务 t 后，对已见任务 0..t 的整体准确率。
- **Accumulated / Average Incremental Acc** = `mean_t(A_t)`（Fly-CL 论文口径，见其 main.py）。
- **Last Acc** = 学完最后一个任务后的整体准确率 `A_{T-1}`。
- **Forgetting** = 各任务「历史最高精度 − 最终精度」的平均（回放/微调类方法才有意义；
  Fly-CL 因累加式闭式解，遗忘≈0，需在报告中解释这一结构性差异）。

## 5. 数据管线（core/data/）

- `ContinualDatasets` 按 `init_cls_num` / `inc_cls_num` / `task_num` 切分类别为若干任务，
  为每个任务建 train/test `DataLoader`；`get_loader(task_idx)` 取用。
- CIFAR-100 走 `binary_cifar100` 分支：直接读 `data_root/cifar-100-python/{train,test}` 的
  pickle（键 `data` 展平为 R(1024)+G(1024)+B(1024)、`fine_labels`）—— 与 torchvision/toronto 原生格式一致。
  
  > 这正是我们 Step 1 里把 fast.ai 镜像的 image-folder 布局**转回**标准 pickle 的原因。
- `class_order` 决定任务内类别构成；**冻结随机种子**即可复现类顺序（Step 4 协议对齐）。
- 变换在 `data.py` 的 `CIFARTransform` 里按 backbone 类型（resnet/vit/alexnet）选择；ViT 走 224 resize。

## 6. 配置规范（YAML）

以 `config/ranpac.yaml`（与 Fly-CL 最同族的 PTM 方法）为模板：

```yaml
init_cls_num: 10          # 第0个任务类别数
inc_cls_num: 10           # 后续每任务类别数
total_cls_num: 100
task_num: 10
init_epoch: 20; epoch: 1  # PTM 方法后续任务常 1 个 epoch
backbone:
  name: vit_pt_imnet_in21k_adapter
  kwargs: { pretrained: true, model_name: vit_base_patch16_224_in21k }
classifier:
  name: RanPAC
  kwargs: { use_RP: true, M: 10000, ... }   # 方法专属超参
```

集成 Fly-CL 需要：

(a) 一个 `classifier.name: FlyCL` 的分类器类；

(b) 一份 `config/flycl.yaml`；
(c) 在 `core/model/__init__.py` 注册。骨干可复用现成 ViT 加载器（或注入我们的冻结特征）。

## 7. 已内置方法（reproduce/ 目录）

经典：LwF, EWC, iCaRL, BiC, LUCIR, WA, DER, OCM, ER-ACE/ER-AML, GPM, TRGP, DMNSP。
PTM：L2P, DualPrompt, CoDA-Prompt, RanPAC, InfLoRA(+opt), SD-LoRA, CL-LoRA, MoE-Adapter, RAPF, PRAKA, DAP。

> **Fly-CL 不在其中** —— 满足「复现 2026 顶会新方法、非官方 repo」的选题要求。
