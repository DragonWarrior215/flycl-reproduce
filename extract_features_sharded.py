"""
Frozen-backbone feature extraction for the Fly-CL reproduction (sharded, resumable).

Datasets (official Fly-CL protocol, datasets/load_dataset.py):
  - cifar100 : cifar-100-python pickles; transform Resize(224,bicubic)+CenterCrop(224)
  - cub      : data/cub/{train,test} ImageFolder (ADAM/PILOT subset, 9430/2358, 200 classes);
               transform Resize(256,bicubic)+CenterCrop(224)
  - vtab     : data/vtab/{train,test} ImageFolder (ADAM/PILOT subset, 1796/8619, 50 classes);
               transform Resize(256,bicubic)+CenterCrop(224)
  All normalized with mean/std 0.5 ("vit" data_augmentation in the official code).

Backbones:
  - augreg_in21k : the paper's ViT-B/16 (timm vit_base_patch16_224, augreg i21k->in1k npz),
                   assets/vit_b16_augreg_in21k_ft_in1k.npz  [default]
  - tv_in1k      : torchvision vit_b_16 IMAGENET1K_V1 (used in the earlier restricted-network
                   reproduction), assets/vit_b_16_torchvision.pth

Usage:
  python extract_features_sharded.py --dataset cifar100 --backbone augreg_in21k
  -> features/{dataset}_vit_b16_{tag}_{train,test}.npz  (tag: augreg / tv)
"""
import os, time, glob, argparse
os.environ["KMP_AFFINITY"]="disabled"; os.environ["OMP_PROC_BIND"]="false"; os.environ["MKL_THREADING_LAYER"]="GNU"
import torch, numpy as np, pickle
from torchvision import transforms
from PIL import Image
torch.set_num_threads(16)

NORM = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
TF_CIFAR = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(), NORM,
])
TF_IMAGEFOLDER = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(), NORM,
])


def build_backbone(name):
    if name == "tv_in1k":
        from torchvision.models import vit_b_16
        m = vit_b_16(weights=None)
        m.load_state_dict(torch.load("assets/vit_b_16_torchvision.pth", map_location="cpu"))
        m.heads = torch.nn.Identity()
    elif name == "augreg_in21k":
        import timm
        from timm.models.vision_transformer import _load_weights
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        _load_weights(m, "assets/vit_b16_augreg_in21k_ft_in1k.npz")
    else:
        raise ValueError(name)
    return m.eval()


def cifar_samples(split):
    """Returns (list of PIL-openable items, labels). Items are ndarray images."""
    with open(f"data/cifar-100-python/{split}","rb") as f:
        d = pickle.load(f, encoding="latin1")
    data = d["data"].reshape(-1,3,32,32).transpose(0,2,3,1)
    return list(data), np.array(d["fine_labels"])


def imagefolder_samples(root):
    """Same class->index mapping as torchvision ImageFolder (sorted dir names)."""
    classes = sorted(e.name for e in os.scandir(root) if e.is_dir())
    paths, labels = [], []
    for idx, c in enumerate(classes):
        for dirpath, _, filenames in sorted(os.walk(os.path.join(root, c))):
            for fn in sorted(filenames):
                paths.append(os.path.join(dirpath, fn)); labels.append(idx)
    return paths, np.array(labels), len(classes)


def load_split(dataset, split):
    if dataset == "cifar100":
        items, labels = cifar_samples(split)
        return items, labels, TF_CIFAR
    root = f"data/{dataset}/{split}"
    paths, labels, ncls = imagefolder_samples(root)
    print(f"{dataset}/{split}: {len(paths)} images, {ncls} classes", flush=True)
    return paths, labels, TF_IMAGEFOLDER


def to_pil(item):
    return (Image.fromarray(item) if isinstance(item, np.ndarray)
            else Image.open(item).convert("RGB"))


SHARD = 5000  # images per shard file

@torch.no_grad()
def extract(dataset, split, model, tf_unused=None, bs=64):
    items, labels, tf = load_split(dataset, split)
    N = len(labels)
    for s0 in range(0, N, SHARD):
        s1 = min(s0+SHARD, N)
        shard_path = f"features/shards/{ARGS.tag}_{dataset}_{split}_{s0:06d}_{s1:06d}.npz"
        if os.path.exists(shard_path):
            print(f"skip {shard_path} (exists)", flush=True); continue
        feats = np.zeros((s1-s0, 768), dtype=np.float32)
        t0=time.time()
        for i in range(s0, s1, bs):
            j1 = min(i+bs, s1)
            batch = torch.stack([tf(to_pil(items[j])) for j in range(i, j1)])
            feats[i-s0:j1-s0] = model(batch).numpy()
        np.savez(shard_path, features=feats, labels=labels[s0:s1])
        print(f"{dataset} {split} shard {s0}-{s1} in {time.time()-t0:.0f}s -> {shard_path}", flush=True)


def combine(dataset, split):
    shards = sorted(glob.glob(f"features/shards/{ARGS.tag}_{dataset}_{split}_*.npz"))
    F = np.concatenate([np.load(s)["features"] for s in shards], axis=0)
    L = np.concatenate([np.load(s)["labels"] for s in shards], axis=0)
    out = f"features/{dataset}_vit_b16_{ARGS.tag}_{split}.npz"
    np.savez_compressed(out, features=F, labels=L)
    print(f"combined {dataset} {split}: {F.shape} -> {out}", flush=True)
    return F.shape


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar100","cub","vtab"], required=True)
    p.add_argument("--backbone", choices=["augreg_in21k","tv_in1k"], default="augreg_in21k")
    ARGS = p.parse_args()
    ARGS.tag = {"augreg_in21k":"augreg", "tv_in1k":"tv"}[ARGS.backbone]
    os.makedirs("features/shards", exist_ok=True)
    model = build_backbone(ARGS.backbone)
    for split in ["train","test"]:
        extract(ARGS.dataset, split, model)
        combine(ARGS.dataset, split)
    print("ALL DONE", flush=True)
