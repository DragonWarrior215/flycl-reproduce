"""
GPU-native Fly-CL analysis on the paper's IN21k augreg ViT-B/16 (features
extracted on GPU, no CPU cache dependency).

- Baselines (§4.2) and ablations (§7): CIFAR-100 only.
- Level-2 brain-inspired experiments (§6.1 decorrelation, §6.2 forgetting
  decomposition, §6.3 CLS-Fly): run on ALL THREE datasets — CIFAR-100,
  CUB-200-2011, VTAB — under each one's official class-incremental protocol.

Run on the GPU server:
  python analysis_gpu.py --weights assets/vit_b16_augreg_in21k.npz \
                         --data-root ./data --out-dir results --fig-dir results
"""
import os, json, time, random, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Per-dataset official protocol. `resize` is the pre-CenterCrop(224) resize:
# CIFAR resizes straight to 224, CUB/VTAB resize to 256 then center-crop.
DATASETS = {
    "cifar100": dict(name="CIFAR-100",    num_classes=100, num_tasks=10, init_cls=10, seed=1993, kind="cifar",       resize=224),
    "cub":      dict(name="CUB-200-2011", num_classes=200, num_tasks=10, init_cls=20, seed=2023, kind="imagefolder", path="cub",  resize=256),
    "vtab":     dict(name="VTAB",         num_classes=50,  num_tasks=5,  init_cls=10, seed=2023, kind="imagefolder", path="vtab", resize=256),
}

# Globals set per-dataset in main(); the fly primitives below read them at runtime.
NUM_CLASSES, NUM_TASKS, SEED = 100, 10, 1993


# ---------------------------------------------------------------- features ----
def _build_backbone(weights_path):
    import timm
    from timm.models.vision_transformer import _load_weights
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    _load_weights(m, weights_path)
    return m.eval().to(DEV)


def _transform(resize):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def extract_features(cfg, weights_path, data_root, cache):
    """Frozen IN21k ViT-B/16 CLS features for a dataset's train/test splits (GPU)."""
    if os.path.exists(cache):
        z = np.load(cache)
        return z["Ftr"], z["Ltr"], z["Fte"], z["Lte"]
    from PIL import Image
    from torchvision import datasets
    from torch.utils.data import DataLoader
    tf = _transform(cfg["resize"])
    m = _build_backbone(weights_path)

    def run_cifar(split):
        with open(os.path.join(data_root, "cifar-100-python", split), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        data = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(d["fine_labels"], dtype=np.int64)
        feats = np.zeros((len(labels), 768), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(labels), 256):
                imgs = torch.stack([tf(Image.fromarray(x)) for x in data[i:i + 256]]).to(DEV)
                feats[i:i + 256] = m(imgs).float().cpu().numpy()
        return feats, labels

    def run_imagefolder(split):
        ds = datasets.ImageFolder(os.path.join(data_root, cfg["path"], split), transform=tf)
        loader = DataLoader(ds, batch_size=256, num_workers=8, shuffle=False)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                feats.append(m(x.to(DEV)).float().cpu())
                labels.append(y)
        return torch.cat(feats).numpy(), torch.cat(labels).numpy().astype(np.int64)

    run = run_cifar if cfg["kind"] == "cifar" else run_imagefolder
    Ftr, Ltr = run("train")
    Fte, Lte = run("test")
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, Ftr=Ftr, Ltr=Ltr, Fte=Fte, Lte=Lte)
    return Ftr, Ltr, Fte, Lte


# ------------------------------------------------------------- fly primitives -
def task_split():
    random.seed(SEED)
    rc = random.sample(list(range(NUM_CLASSES)), NUM_CLASSES)
    cpt = NUM_CLASSES // NUM_TASKS
    return [rc[i * cpt:(i + 1) * cpt] for i in range(NUM_TASKS)]


def build_W(M, d, s, seed):
    g = torch.Generator().manual_seed(seed)
    W = torch.zeros(M, d)
    for r in range(M):
        cols = torch.randperm(d, generator=g)[:s]
        W[r, cols] = torch.randn(s, generator=g)
    return W.to(DEV)


def pwta(Fx, W, k):
    z = Fx @ W.T
    v, idx = z.topk(k, dim=1)
    o = torch.zeros_like(z)
    o.scatter_(1, idx, v)
    return o


def onehot(y, C):
    o = torch.zeros(y.shape[0], C, device=y.device)
    o.scatter_(1, y.long().view(-1, 1), 1.0)
    return o


def gcv_ridge(Phi, Y, lo=6, hi=10):
    U, S, _ = torch.linalg.svd(Phi, full_matrices=False)
    Ss = S ** 2
    UTY = U.T @ Y
    rid = torch.tensor(10.0 ** np.arange(lo, hi), device=Phi.device)
    n = Phi.shape[0]
    best, br = 1e30, rid[0]
    for r in rid:
        diag = Ss / (Ss + r)
        df = diag.sum()
        Yh = U @ (diag[:, None] * UTY)
        g = (torch.norm(Y - Yh) ** 2 / n) / (1 - df / n) ** 2
        if g.item() < best:
            best, br = g.item(), r
    return br


def chol_solve(G, Q, r):
    A = G.clone(); A.diagonal().add_(r)
    return torch.cholesky_solve(Q, torch.linalg.cholesky(A))


def solve_ridge(G, Q, r):
    A = G.clone(); A.diagonal().add_(r)
    return torch.linalg.solve(A, Q)


def acc_metrics(acc):
    # acc[st][k] is the accuracy on task st at stage (st + k); A_t = mean over
    # seen tasks of their stage-t accuracy.
    return [round(sum(acc[st][j - st] for st in range(j + 1)) / (j + 1), 2) for j in range(NUM_TASKS)]


# ------------------------------------------------------------ Fly-CL / baselines
def flycl(Ftr, Ltr, Fte, Lte, M=10000, s=300, rho=0.3, keep_state=False):
    torch.manual_seed(SEED)
    W = build_W(M, Ftr.shape[1], s, SEED)
    k = int(M * rho)
    tc = task_split()
    Q = torch.zeros(M, NUM_CLASSES, device=DEV)
    G = torch.zeros(M, M, device=DEV)
    acc = {t: [] for t in range(NUM_TASKS)}
    fresh, tt, Wo = {}, [], None
    for task in range(NUM_TASKS):
        mtr = torch.isin(Ltr, torch.tensor(tc[task], device=DEV))
        t0 = time.time()
        Phi = pwta(Ftr[mtr], W, k)
        Y = onehot(Ltr[mtr], NUM_CLASSES)
        Q += Phi.T @ Y
        G += Phi.T @ Phi
        Wo = chol_solve(G, Q, gcv_ridge(Phi, Y))
        tt.append(time.time() - t0)
        for st in range(task + 1):
            mte = torch.isin(Lte, torch.tensor(tc[st], device=DEV))
            a = ((pwta(Fte[mte], W, k) @ Wo).argmax(1) == Lte[mte]).float().mean().item() * 100
            acc[st].append(a)
            if st == task:
                fresh[st] = a
    A = acc_metrics(acc)
    out = dict(A_t=A, accumulated=round(float(np.mean(A)), 2), last=A[-1],
               avg_train_s=round(float(np.mean(tt)), 2), fresh=fresh)
    if keep_state:
        out["_state"] = dict(W=W, k=k, Wo=Wo, tc=tc)
    return out


def ranpac(Ftr, Ltr, Fte, Lte, M=10000):
    torch.manual_seed(SEED)
    Wr = torch.randn(Ftr.shape[1], M, device=DEV)
    tc = task_split()
    proj = lambda Fx: torch.relu(Fx @ Wr)
    Q = torch.zeros(M, NUM_CLASSES, device=DEV)
    G = torch.zeros(M, M, device=DEV)
    acc = {t: [] for t in range(NUM_TASKS)}
    for task in range(NUM_TASKS):
        mtr = torch.isin(Ltr, torch.tensor(tc[task], device=DEV))
        Phi = proj(Ftr[mtr]); Y = onehot(Ltr[mtr], NUM_CLASSES)
        Q += Phi.T @ Y; G += Phi.T @ Phi
        rid = 10.0 ** np.arange(-8, 9); nv = int(Phi.shape[0] * 0.8)
        Qv, Gv = Phi[:nv].T @ Y[:nv], Phi[:nv].T @ Phi[:nv]
        best, br = 1e30, rid[0]
        for r in rid:
            loss = F.mse_loss(Phi[nv:] @ solve_ridge(Gv, Qv, r), Y[nv:]).item()
            if loss < best:
                best, br = loss, r
        Wo = solve_ridge(G, Q, br)
        for st in range(task + 1):
            mte = torch.isin(Lte, torch.tensor(tc[st], device=DEV))
            acc[st].append((proj(Fte[mte]) @ Wo).argmax(1).eq(Lte[mte]).float().mean().item() * 100)
    A = acc_metrics(acc)
    return dict(A_t=A, accumulated=round(float(np.mean(A)), 2), last=A[-1])


def ncm(Ftr, Ltr, Fte, Lte):
    tc = task_split()
    protos = torch.zeros(NUM_CLASSES, Ftr.shape[1], device=DEV)
    seen = []
    acc = {t: [] for t in range(NUM_TASKS)}
    for task in range(NUM_TASKS):
        for c in tc[task]:
            protos[c] = Ftr[Ltr == c].mean(0)
        seen += tc[task]; st_t = torch.tensor(seen, device=DEV)
        for st in range(task + 1):
            mte = torch.isin(Lte, torch.tensor(tc[st], device=DEV))
            dcol = torch.cdist(Fte[mte], protos[st_t])
            acc[st].append((st_t[dcol.argmin(1)] == Lte[mte]).float().mean().item() * 100)
    A = acc_metrics(acc)
    return dict(A_t=A, accumulated=round(float(np.mean(A)), 2), last=A[-1])


def ablations(Ftr, Ltr, Fte, Lte):
    out = {"M": {}, "rho": {}, "s": {}}
    for M in [1000, 2000, 5000, 10000, 20000]:
        out["M"][str(M)] = flycl(Ftr, Ltr, Fte, Lte, M=M, s=300, rho=0.3)["accumulated"]
        print(f"  ablation M={M}: {out['M'][str(M)]}", flush=True)
    for rho in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        out["rho"][str(rho)] = flycl(Ftr, Ltr, Fte, Lte, M=5000, s=300, rho=rho)["accumulated"]
        print(f"  ablation rho={rho}: {out['rho'][str(rho)]}", flush=True)
    for s in [8, 16, 32, 100, 300, 768]:
        out["s"][str(s)] = flycl(Ftr, Ltr, Fte, Lte, M=5000, s=s, rho=0.3)["accumulated"]
        print(f"  ablation s={s}: {out['s'][str(s)]}", flush=True)
    return out


# ---------------------------------------------- §6.1 decorrelation (Level-2) ---
def mean_offdiag_corr(X):
    X = X - X.mean(0, keepdim=True)
    Xn = X / (X.std(0, keepdim=True) + 1e-8)
    C = (Xn.T @ Xn) / (X.shape[0] - 1)
    n = C.shape[0]
    return ((C.abs().sum() - C.abs().diagonal().sum()) / (n * n - n)).item()


def decorrelation(Ftr, Ltr):
    torch.manual_seed(SEED)
    idx = []
    for c in range(NUM_CLASSES):
        ci = torch.where(Ltr == c)[0]
        idx.append(ci[torch.randperm(len(ci))[:20]])
    idx = torch.cat(idx)
    Fsub = Ftr[idx]; labs = Ltr[idx]
    d, Mp = Fsub.shape[1], 5000
    raw = mean_offdiag_corr(Fsub)
    dense = mean_offdiag_corr(Fsub @ torch.randn(d, Mp, device=DEV))
    Ws = build_W(Mp, d, 300, SEED); k = int(Mp * 0.3)
    Zk = pwta(Fsub, Ws, k)
    wta = mean_offdiag_corr(Zk)

    def class_sim(Z):
        Zc = F.normalize(Z, dim=1)
        within = [(Zc[labs == c] @ Zc[labs == c].T).mean().item()
                  for c in range(NUM_CLASSES) if (labs == c).sum() > 1]
        cmeans = F.normalize(torch.stack([Zc[labs == c].mean(0) for c in range(NUM_CLASSES)]), dim=1)
        S = cmeans @ cmeans.T
        between = (S.sum() - S.diagonal().sum()).item() / (NUM_CLASSES * (NUM_CLASSES - 1))
        return round(float(np.mean(within)), 3), round(between, 3)

    raw_w, raw_b = class_sim(Fsub)
    k_w, k_b = class_sim(Zk)
    return {
        "offdiag_corr": {"raw": round(raw, 4), "dense_proj": round(dense, 4), "proj_wta": round(wta, 4)},
        "corr_reduction_vs_dense_pct": round((1 - wta / dense) * 100, 1),
        "class_similarity": {"raw": {"within": raw_w, "between": raw_b},
                              "kwta": {"within": k_w, "between": k_b}},
    }


# ---------------------------------------------- §6.2 forgetting (Level-2) ------
def forgetting(Ftr, Ltr, Fte, Lte, M=10000):
    st = flycl(Ftr, Ltr, Fte, Lte, M=M, s=300, rho=0.3, keep_state=True)
    W, k, Wo, tc = (st["_state"][x] for x in ("W", "k", "Wo", "tc"))
    t0 = torch.tensor(tc[0], device=DEV)
    mte = torch.isin(Lte, t0)
    logits = pwta(Fte[mte], W, k) @ Wo
    fresh = st["fresh"][0]
    all_c = (logits.argmax(1) == Lte[mte]).float().mean().item() * 100
    restr = (t0[logits[:, t0].argmax(1)] == Lte[mte]).float().mean().item() * 100
    return {
        "task0_fresh_acc": round(fresh, 1),
        "task0_final_restricted": round(restr, 1),      # scored only within task-0 classes
        "task0_final_all_classes": round(all_c, 1),     # scored over all classes (reported "final")
        "measured_forgetting": round(fresh - all_c, 1),
        "true_forgetting": round(fresh - restr, 1),      # genuine representational loss
        "label_space_growth": round(restr - all_c, 1),   # rest, from harder discrimination
    }


# ---------------------------------------------- §6.3 CLS-Fly (Level-2) ---------
def clsfly_run(Ftr, Ltr, Fte, Lte, M, betas):
    torch.manual_seed(SEED)
    W = build_W(M, Ftr.shape[1], 300, SEED); k = int(M * 0.3)
    tc = task_split()
    Q = torch.zeros(M, NUM_CLASSES, device=DEV)
    G = torch.zeros(M, M, device=DEV)
    proto_sum = torch.zeros(NUM_CLASSES, M, device=DEV)
    per_task = {b: {t: [] for t in range(NUM_TASKS)} for b in betas}
    zscore = lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
    for task in range(NUM_TASKS):
        cls = tc[task]
        mtr = torch.isin(Ltr, torch.tensor(cls, device=DEV))
        Phi = pwta(Ftr[mtr], W, k); Y = onehot(Ltr[mtr], NUM_CLASSES)
        Q += Phi.T @ Y; G += Phi.T @ Phi
        for c in cls:
            proto_sum[c] = pwta(Ftr[Ltr == c], W, k).mean(0)
        Wo = chol_solve(G, Q, gcv_ridge(Phi, Y))
        protos = F.normalize(proto_sum, dim=1)
        for st in range(task + 1):
            mte = torch.isin(Lte, torch.tensor(tc[st], device=DEV))
            Pt = pwta(Fte[mte], W, k)
            ridge_s = zscore(Pt @ Wo)
            proto_s = zscore(F.normalize(Pt, dim=1) @ protos.T)
            lab = Lte[mte]
            for b in betas:
                fused = (1 - b) * ridge_s + b * proto_s
                per_task[b][st].append((fused.argmax(1) == lab).float().mean().item() * 100)
    return {str(b): round(float(np.mean(acc_metrics(per_task[b]))), 2) for b in betas}


def level2(Ftr, Ltr, Fte, Lte):
    dec = decorrelation(Ftr, Ltr)
    print("  decorrelation:", dec["offdiag_corr"], "reduction%", dec["corr_reduction_vs_dense_pct"], flush=True)
    frg = forgetting(Ftr, Ltr, Fte, Lte, M=10000)
    print("  forgetting:", frg, flush=True)
    big = clsfly_run(Ftr, Ltr, Fte, Lte, M=5000, betas=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    print("  clsfly M=5000:", big, flush=True)
    small = {str(M): clsfly_run(Ftr, Ltr, Fte, Lte, M=M, betas=[0.0, 0.1, 0.2, 0.3, 0.5])
             for M in [300, 500, 1000]}
    for M, r in small.items():
        print(f"  clsfly M={M}:", r, flush=True)
    return {"decorrelation": dec, "forgetting": frg, "clsfly_M5000": big, "clsfly_smallM": small}


# ------------------------------------------------------------------ figures ---
COLORS = {"cifar100": "#3B6FB6", "cub": "#d98c5f", "vtab": "#5a9e6f"}


def make_figures(fig_dir, base, abl, level2_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dss = list(level2_all.keys())
    names = {"cifar100": "CIFAR-100", "cub": "CUB-200", "vtab": "VTAB"}

    # ablation.png (CIFAR-100)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, key, xl in zip(ax, ["M", "rho", "s"], ["expand dim M", "coding level ρ", "synaptic degree s"]):
        xs = list(abl[key].keys()); ys = [abl[key][x] for x in xs]
        a.plot(range(len(xs)), ys, "o-", color="#3B6FB6")
        a.set_xticks(range(len(xs))); a.set_xticklabels(xs)
        a.set_xlabel(xl); a.set_ylabel("Accumulated Ā (%)"); a.grid(alpha=.3)
    fig.suptitle("Fly-CL ablations (CIFAR-100, IN21k ViT-B/16)")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/ablation.png", dpi=130); plt.close(fig)

    # decorrelation_analysis.png — off-diag corr across the three datasets
    fig, ax = plt.subplots(figsize=(7.5, 4))
    stages = ["raw", "dense_proj", "proj_wta"]; labels = ["raw ViT", "dense proj", "proj + k-WTA"]
    x = np.arange(len(dss)); w = 0.25
    for j, (stg, col) in enumerate(zip(stages, ["#9aa7b3", "#d98c5f", "#3B6FB6"])):
        vals = [level2_all[ds]["decorrelation"]["offdiag_corr"][stg] for ds in dss]
        ax.bar(x + (j - 1) * w, vals, w, label=labels[j], color=col)
    ax.set_xticks(x); ax.set_xticklabels([names[d] for d in dss])
    ax.set_ylabel("mean |off-diag corr|"); ax.grid(alpha=.3, axis="y"); ax.legend()
    ax.set_title("k-WTA decorrelation across datasets (IN21k ViT-B/16)")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/decorrelation_analysis.png", dpi=130); plt.close(fig)

    # clsfly_extension.png — beta sweeps at well-determined (M=5000) and small (M=1000) M
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for panel, (Mkey, title) in enumerate([("M5000", "Well-determined (M=5000)"),
                                            ("small1000", "Under-determined (M=1000)")]):
        for ds in dss:
            series = level2_all[ds]["clsfly_M5000"] if Mkey == "M5000" else level2_all[ds]["clsfly_smallM"]["1000"]
            bx = [float(b) for b in series]; by = [series[b] for b in series]
            ax[panel].plot(bx, by, "o-", label=names[ds], color=COLORS[ds])
        ax[panel].set_xlabel("fusion β"); ax[panel].set_ylabel("Accumulated Ā (%)")
        ax[panel].set_title(title); ax[panel].grid(alpha=.3); ax[panel].legend()
    fig.suptitle("CLS-Fly: slow ridge + fast prototype across datasets")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/clsfly_extension.png", dpi=130); plt.close(fig)

    # accuracy_curves.png + comparison_bars.png (CIFAR baselines)
    fig, ax = plt.subplots(figsize=(6, 4))
    for nm, c in [("FlyCL", "#3B6FB6"), ("RanPAC", "#d98c5f"), ("NCM", "#9aa7b3")]:
        ax.plot(range(1, len(base[nm]["A_t"]) + 1), base[nm]["A_t"], "o-", label=nm, color=c)
    ax.set_xlabel("stage t"); ax.set_ylabel("A_t (%)"); ax.set_title("Per-stage accuracy (CIFAR-100, IN21k)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/accuracy_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    nm = ["FlyCL", "RanPAC", "NCM"]
    ax.bar(nm, [base[n]["accumulated"] for n in nm], color=["#3B6FB6", "#d98c5f", "#9aa7b3"])
    for i, n in enumerate(nm):
        ax.text(i, base[n]["accumulated"] + .3, f'{base[n]["accumulated"]}', ha="center")
    ax.set_ylabel("Accumulated Ā (%)"); ax.set_title("Method comparison (CIFAR-100, IN21k)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/comparison_bars.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------- main --
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="assets/vit_b16_augreg_in21k.npz")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--fig-dir", default="results")
    ap.add_argument("--cache-dir", default="./.feat_cache")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={DEV}", flush=True)
    t_all = time.time()

    base = abl = None
    level2_all = {}
    for ds, cfg in DATASETS.items():
        NUM_CLASSES, NUM_TASKS, SEED = cfg["num_classes"], cfg["num_tasks"], cfg["seed"]
        print(f"\n########## {cfg['name']} "
              f"({NUM_CLASSES} classes, {NUM_TASKS} tasks, seed {SEED}) ##########", flush=True)
        t0 = time.time()
        cache = os.path.join(args.cache_dir, f"{ds}_in21k.npz")
        Ftr, Ltr, Fte, Lte = extract_features(cfg, args.weights, args.data_root, cache)
        Ftr = torch.tensor(Ftr, device=DEV); Ltr = torch.tensor(Ltr, device=DEV)
        Fte = torch.tensor(Fte, device=DEV); Lte = torch.tensor(Lte, device=DEV)
        print(f"features: train {tuple(Ftr.shape)} test {tuple(Fte.shape)} ({time.time()-t0:.0f}s)", flush=True)

        if ds == "cifar100":
            print("== baselines (CIFAR-100) ==", flush=True)
            base = {"FlyCL": flycl(Ftr, Ltr, Fte, Lte, M=10000),
                    "RanPAC": ranpac(Ftr, Ltr, Fte, Lte, M=10000),
                    "NCM": ncm(Ftr, Ltr, Fte, Lte)}
            for nm in base:
                print(f"  {nm}: {base[nm]['accumulated']} / {base[nm]['last']}", flush=True)
            json.dump(base, open(f"{args.out_dir}/all_methods.json", "w"), indent=1)
            print("== ablations (CIFAR-100) ==", flush=True)
            abl = ablations(Ftr, Ltr, Fte, Lte)
            json.dump(abl, open(f"{args.out_dir}/ablation_all.json", "w"), indent=2)

        print(f"== Level-2 ({cfg['name']}) ==", flush=True)
        level2_all[ds] = level2(Ftr, Ltr, Fte, Lte)
        level2_all[ds]["_meta"] = {"dataset": cfg["name"], "num_classes": NUM_CLASSES,
                                   "num_tasks": NUM_TASKS, "init_cls": cfg["init_cls"], "seed": SEED}
        del Ftr, Ltr, Fte, Lte
        torch.cuda.empty_cache()

    # Level-2 across all three datasets (combined) + CIFAR-named singles (back-compat)
    json.dump(level2_all, open(f"{args.out_dir}/level2_3datasets.json", "w"), indent=2)
    c = level2_all["cifar100"]
    json.dump(c["decorrelation"], open(f"{args.out_dir}/decorrelation_analysis.json", "w"), indent=2)
    json.dump(c["forgetting"], open(f"{args.out_dir}/forgetting_decomposition.json", "w"), indent=2)
    json.dump(c["clsfly_M5000"], open(f"{args.out_dir}/clsfly.json", "w"), indent=1)
    json.dump(c["clsfly_smallM"], open(f"{args.out_dir}/clsfly_smallM.json", "w"), indent=2)

    print("\n== figures ==", flush=True)
    make_figures(args.fig_dir, base, abl, level2_all)
    print(f"\nALL DONE ({time.time()-t_all:.0f}s)", flush=True)
