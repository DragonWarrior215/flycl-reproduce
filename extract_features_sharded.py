import os, time, glob
os.environ["KMP_AFFINITY"]="disabled"; os.environ["OMP_PROC_BIND"]="false"; os.environ["MKL_THREADING_LAYER"]="GNU"
import torch, numpy as np, pickle
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image
torch.set_num_threads(16)

tf = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
class FE(torch.nn.Module):
    def __init__(s, wp):
        super().__init__()
        m = vit_b_16(weights=None); m.load_state_dict(torch.load(wp, map_location="cpu"))
        m.heads = torch.nn.Identity(); s.m = m.eval()
    @torch.no_grad()
    def forward(s, x): return s.m(x)

def load_split(split):
    with open(f"data/cifar-100-python/{split}","rb") as f:
        d = pickle.load(f, encoding="latin1")
    data = d["data"].reshape(-1,3,32,32).transpose(0,2,3,1)
    return data, np.array(d["fine_labels"])

SHARD = 5000  # images per shard file
os.makedirs("features/shards", exist_ok=True)

def extract(split, fe, bs=64):
    imgs, labels = load_split(split)
    N = len(labels)
    for s0 in range(0, N, SHARD):
        s1 = min(s0+SHARD, N)
        shard_path = f"features/shards/{split}_{s0:06d}_{s1:06d}.npz"
        if os.path.exists(shard_path):
            print(f"skip {shard_path} (exists)", flush=True); continue
        feats = np.zeros((s1-s0, 768), dtype=np.float32)
        t0=time.time()
        for i in range(s0, s1, bs):
            j1 = min(i+bs, s1)
            batch = torch.stack([tf(Image.fromarray(imgs[j])) for j in range(i, j1)])
            with torch.no_grad(): feats[i-s0:j1-s0] = fe(batch).numpy()
        np.savez(shard_path, features=feats, labels=labels[s0:s1])
        print(f"{split} shard {s0}-{s1} in {time.time()-t0:.0f}s -> {shard_path}", flush=True)

def combine(split):
    shards = sorted(glob.glob(f"features/shards/{split}_*.npz"))
    F = np.concatenate([np.load(s)["features"] for s in shards], axis=0)
    L = np.concatenate([np.load(s)["labels"] for s in shards], axis=0)
    np.savez_compressed(f"features/cifar100_vit_b16_tv_{split}.npz", features=F, labels=L)
    print(f"combined {split}: {F.shape} -> features/cifar100_vit_b16_tv_{split}.npz", flush=True)
    return F.shape

if __name__ == "__main__":
    fe = FE("assets/vit_b_16_torchvision.pth")
    for split in ["train","test"]:
        extract(split, fe)
        combine(split)
    print("ALL DONE", flush=True)
