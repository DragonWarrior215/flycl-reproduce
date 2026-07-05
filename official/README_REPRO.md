# 官方框架复现记录（2026-07-05）

本目录是 Fly-CL **官方仓库**的复现快照与运行入口，对应主 README §1.1 的结果。

## 来源与改动

- `Fly-CL/`：克隆自 <https://github.com/gfyddha/Fly-CL>，上游 commit
  `193b1b80707d59d3fbe2e3342155aabe4e155396`（"Update README"），已去除 `.git`。
- **唯一代码改动**：`Fly-CL/models/load_model.py` 新增 `vit_base_patch16_224_in21k`
  分支，加载 timm `vit_base_patch16_224.augreg_in21k`——即作者
  `pretrained_model/download.sh` 所指定的权重（augreg IN21k，未经 IN1k 微调）。
  原有 `vit_base_patch16_224`（`pretrained=True` → timm 默认 `augreg2_in21k_ft_in1k`）
  分支未动，两种权重的结果都在 `../results/official_framework/`。

## 运行环境（AutoDL RTX 4090）

- Python 3.10 / torch 2.1.2+cu118 / torchvision 0.16.2 / timm 0.9.16 / numpy 1.26 / scipy
- 与官方 README 建议的 torch 1.13.1 不同，但 Fly-CL 全程冻结骨干 + 闭式解，
  未见任何版本相关问题。

## 数据布局（`--root ../data`，即 `Fly-CL` 同级的 `data/`）

```
data/
  cifar-100-python/        # 官方 tar.gz 解包（md5 eb9058c3a382ffc7106e4002c42a8d85）
  cub/{train,test}/        # ImageFolder，200 类
  vtab/{train,test}/       # ImageFolder，50 类
```

CIFAR-100 必须是**官方原版** tar.gz（本仓库 `data/cifar-100-python-official.tar.gz`），
否则 torchvision 校验失败会转入慢速重下载。

## 权重

- `assets/vit_b16_augreg_in21k.safetensors`：timm `vit_base_patch16_224.augreg_in21k`
  的 `model.safetensors`（复现论文数值所用）。离线加载方式：
  `timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=True,
  pretrained_cfg_overlay=dict(file="assets/vit_b16_augreg_in21k.safetensors"), num_classes=0)`；
  或放入 HF 缓存后设 `HF_HUB_OFFLINE=1`（服务器采用后者）。
- 国内网络从 hf-mirror.com 下载：`HF_ENDPOINT=https://hf-mirror.com hf download
  timm/vit_base_patch16_224.augreg_in21k`。

## 运行

```bash
bash run_all.sh          # timm 默认权重（augreg2_in21k_ft_in1k）三数据集顺序执行
bash run_all_in21k.sh    # download.sh 对应的 IN21k 权重（复现论文数值用这个）
```

日志输出到 `logs/{cifar100,cub,vtab}[_in21k].log`（本仓库已归档至
`results/official_framework/`）。超参与官方 `scripts/test_*.sh` 完全一致，仅
`--gpu` 改为 0（单卡服务器）。

## 结果摘要

| 数据集 | 论文 Ā | IN21k 复现 | timm 默认 |
| --- | --- | --- | --- |
| CIFAR-100 | 93.89 ± 0.12 | 93.88 | 93.20 |
| CUB-200-2011 | 93.84 ± 0.18 | 93.84 | 91.92 |
| VTAB | 96.54 ± 0.38 | 95.73 | 94.84 |

详见 `../results/official_framework/summary.json`。
