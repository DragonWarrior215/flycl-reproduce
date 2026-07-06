"""
GPU-native Fly-CL analysis: baselines (§1.2), ablations (§8) and the Level-2
brain-inspired experiments (§10.1 decorrelation, §10.2 forgetting decomposition,
§10.3 CLS-Fly), all on the paper's IN21k augreg ViT-B/16, features extracted on
GPU. Replaces the old CPU cached-feature driver -- no CPU feature extraction.

CIFAR-100, official class-incremental protocol (100 classes, 10 tasks, seed 1993,
random-permutation task split). Fly-CL: sparse random projection (s non-zeros/row)
+ k-WTA (keep ceil(rho*M)) + accumulated ridge with GCV lambda; predict argmax.

Run on the GPU server:
  python analysis_gpu.py --weights assets/vit_b16_augreg_in21k.npz \
                         --data-root ./data --out-dir results --fig-dir results
"""
import os, json, time, random, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES, NUM_TASKS, SEED = 100, 10, 1993


# ---------------------------------------------------------------- features ----
def extract_cifar_features(weights_path, data_root, cache):
    """Frozen IN21k ViT-B/16 CLS features for CIFAR-100 train/test, extracted on GPU.
    CIFAR transform (paper): Resize(224, BICUBIC) + CenterCrop(224) + Normalize(0.5)."""
    if os.path.exists(cache):
        z = np.load(cache)
        return z["Ftr"], z["Ltr"], z["Fte"], z["Lte"]
    import timm
    from timm.models.vision_transformer import _load_weights
    from torchvision import transforms
    from PIL import Image

    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    _load_weights(m, weights_path)
    m.eval().to(DEV)

    def load_split(name):
        with open(os.path.join(data_root, "cifar-100-python", name), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        data = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC uint8
        labels = np.array(d["fine_labels"], dtype=np.int64)
        feats = np.zeros((len(labels), 768), dtype=np.float32)
        bs = 256
        with torch.no_grad():
            for i in range(0, len(labels), bs):
                imgs = torch.stack([tf(Image.fromarray(x)) for x in data[i:i + bs]]).to(DEV)
                feats[i:i + bs] = m(imgs).float().cpu().numpy()
                if i % (bs * 20) == 0:
                    print(f"  {name}: {i}/{len(labels)}", flush=True)
        return feats, labels

    Ftr, Ltr = load_split("train")
    Fte, Lte = load_split("test")
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, Ftr=Ftr, Ltr=Ltr, Fte=Fte, Lte=Lte)
    return Ftr, Ltr, Fte, Lte


# ------------------------------------------------------------- fly primitives -
def task_split(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS, seed=SEED):
    random.seed(seed)
    rc = random.sample(list(range(num_classes)), num_classes)
    cpt = num_classes // num_tasks
    return [rc[i * cpt:(i + 1) * cpt] for i in range(num_tasks)]


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
    A = G.clone()
    A.diagonal().add_(r)
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(Q, L)


def solve_ridge(G, Q, r):
    A = G.clone()
    A.diagonal().add_(r)
    return torch.linalg.solve(A, Q)


# ------------------------------------------------------------ Fly-CL / baselines
def acc_metrics(acc):
    # acc[st] is appended once per task >= st, so acc[st][k] is the accuracy on
    # task st at stage (st + k). A_t = mean over seen tasks of their stage-t acc.
    A = [round(sum(acc[st][j - st] for st in range(j + 1)) / (j + 1), 2) for j in range(NUM_TASKS)]
    return A


def flycl(Ftr, Ltr, Fte, Lte, M=10000, s=300, rho=0.3, seed=SEED, keep_state=False):
    torch.manual_seed(seed)
    d = Ftr.shape[1]
    W = build_W(M, d, s, seed)
    k = int(M * rho)
    tc = task_split()
    Q = torch.zeros(M, NUM_CLASSES, device=DEV)
    G = torch.zeros(M, M, device=DEV)
    acc = {t: [] for t in range(NUM_TASKS)}
    tt = []
    fresh = {}
    Wo = None
    for task in range(NUM_TASKS):
        cls = torch.tensor(tc[task], device=DEV)
        mtr = torch.isin(Ltr, cls)
        t0 = time.time()
        Phi = pwta(Ftr[mtr], W, k)
        Y = onehot(Ltr[mtr], NUM_CLASSES)
        Q += Phi.T @ Y
        G += Phi.T @ Phi
        r = gcv_ridge(Phi, Y)
        Wo = chol_solve(G, Q, r)
        tt.append(time.time() - t0)
        for st in range(task + 1):
            mte = torch.isin(Lte, torch.tensor(tc[st], device=DEV))
            pred = (pwta(Fte[mte], W, k) @ Wo).argmax(1)
            a = (pred == Lte[mte]).float().mean().item() * 100
            acc[st].append(a)
            if st == task:  # accuracy on task st right after it was learned
                fresh[st] = a
    A = acc_metrics(acc)
    out = dict(A_t=A, accumulated=round(float(np.mean(A)), 2), last=A[-1],
               avg_train_s=round(float(np.mean(tt)), 2), fresh=fresh)
    if keep_state:
        out["_state"] = dict(W=W, k=k, Wo=Wo, tc=tc, acc=acc)
    return out


def ranpac(Ftr, Ltr, Fte, Lte, M=10000, seed=SEED):
    torch.manual_seed(seed)
    d = Ftr.shape[1]
    Wr = torch.randn(d, M, device=DEV)
    tc = task_split()
    proj = lambda Fx: torch.relu(Fx @ Wr)
    Q = torch.zeros(M, NUM_CLASSES, device=DEV)
    G = torch.zeros(M, M, device=DEV)
    acc = {t: [] for t in range(NUM_TASKS)}
    for task in range(NUM_TASKS):
        cls = torch.tensor(tc[task], device=DEV)
        mtr = torch.isin(Ltr, cls)
        Phi = proj(Ftr[mtr])
        Y = onehot(Ltr[mtr], NUM_CLASSES)
        Q += Phi.T @ Y
        G += Phi.T @ Phi
        rid = 10.0 ** np.arange(-8, 9)
        nv = int(Phi.shape[0] * 0.8)
        Qv, Gv = Phi[:nv].T @ Y[:nv], Phi[:nv].T @ Phi[:nv]
        best, br = 1e30, rid[0]
        for r in rid:
            Wo = solve_ridge(Gv, Qv, r)
            loss = F.mse_loss(Phi[nv:] @ Wo, Y[nv:]).item()
            if loss < best:
                best, br = loss, r
        Wo = solve_ridge(G, Q, br)
        for st in range(task + 1):
            mte = torch.isin(Lte, torch.tensor(tc[st], device=DEV))
            acc[st].append((proj(Fte[mte]) @ Wo).argmax(1).eq(Lte[mte]).float().mean().item() * 100)
    A = acc_metrics(acc)
    return dict(A_t=A, accumulated=round(float(np.mean(A)), 2), last=A[-1])


def ncm(Ftr, Ltr, Fte, Lte, seed=SEED):
    tc = task_split()
    protos = torch.zeros(NUM_CLASSES, Ftr.shape[1], device=DEV)
    seen = []
    acc = {t: [] for t in range(NUM_TASKS)}
    for task in range(NUM_TASKS):
        for c in tc[task]:
            protos[c] = Ftr[Ltr == c].mean(0)
        seen += tc[task]
        st_t = torch.tensor(seen, device=DEV)
        for st in range(task + 1):
            mte = torch.isin(Lte, torch.tensor(tc[st], device=DEV))
            dcol = torch.cdist(Fte[mte], protos[st_t])
            acc[st].append((st_t[dcol.argmin(1)] == Lte[mte]).float().mean().item() * 100)
    A = acc_metrics(acc)
    return dict(A_t=A, accumulated=round(float(np.mean(A)), 2), last=A[-1])


# ------------------------------------------------------------------ ablations --
def ablations(Ftr, Ltr, Fte, Lte):
    Ms = [1000, 2000, 5000, 10000, 20000]
    rhos = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    ss = [8, 16, 32, 100, 300, 768]
    out = {"M": {}, "rho": {}, "s": {}}
    for M in Ms:
        out["M"][str(M)] = flycl(Ftr, Ltr, Fte, Lte, M=M, s=300, rho=0.3)["accumulated"]
        print(f"  ablation M={M}: {out['M'][str(M)]}", flush=True)
    for rho in rhos:
        out["rho"][str(rho)] = flycl(Ftr, Ltr, Fte, Lte, M=5000, s=300, rho=rho)["accumulated"]
        print(f"  ablation rho={rho}: {out['rho'][str(rho)]}", flush=True)
    for s in ss:
        out["s"][str(s)] = flycl(Ftr, Ltr, Fte, Lte, M=5000, s=s, rho=0.3)["accumulated"]
        print(f"  ablation s={s}: {out['s'][str(s)]}", flush=True)
    return out


# --------------------------------------------------- §10.1 decorrelation -------
def mean_offdiag_corr(X):
    X = X - X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True) + 1e-8
    Xn = X / sd
    C = (Xn.T @ Xn) / (X.shape[0] - 1)
    n = C.shape[0]
    off = C.abs().sum() - C.abs().diagonal().sum()
    return (off / (n * n - n)).item()


def decorrelation(Ftr, Ltr, seed=SEED):
    torch.manual_seed(seed)
    # class-balanced subset: 20 per class -> 2000
    idx = []
    for c in range(NUM_CLASSES):
        ci = torch.where(Ltr == c)[0]
        idx.append(ci[torch.randperm(len(ci))[:20]])
    idx = torch.cat(idx)
    Fsub = Ftr[idx]
    d = Fsub.shape[1]
    Mp = 5000
    raw = mean_offdiag_corr(Fsub)
    Wd = torch.randn(d, Mp, device=DEV)
    dense = mean_offdiag_corr(Fsub @ Wd)
    Ws = build_W(Mp, d, 300, seed)
    k = int(Mp * 0.3)
    wta = mean_offdiag_corr(pwta(Fsub, Ws, k))
    # class similarity (cosine) raw vs kWTA
    def class_sim(Z):
        Zc = F.normalize(Z, dim=1)
        within, between = [], []
        labs = Ltr[idx]
        for c in range(NUM_CLASSES):
            m = labs == c
            zc = Zc[m]
            if len(zc) > 1:
                within.append((zc @ zc.T).mean().item())
        # between: sample class-mean cosines
        cmeans = F.normalize(torch.stack([Zc[labs == c].mean(0) for c in range(NUM_CLASSES)]), dim=1)
        S = cmeans @ cmeans.T
        between = (S.sum() - S.diagonal().sum()).item() / (NUM_CLASSES * (NUM_CLASSES - 1))
        return round(float(np.mean(within)), 3), round(between, 3)
    raw_w, raw_b = class_sim(Fsub)
    Zk = pwta(Fsub, Ws, k)
    k_w, k_b = class_sim(Zk)
    return {
        "offdiag_corr": {"raw": round(raw, 4), "dense_proj": round(dense, 4), "proj_wta": round(wta, 4)},
        "corr_reduction_vs_dense_pct": round((1 - wta / dense) * 100, 1),
        "population_sparsity": {"raw": 1.0, "kwta": 0.3},
        "class_similarity": {"raw": {"within": raw_w, "between": raw_b},
                              "kwta": {"within": k_w, "between": k_b}},
    }


# --------------------------------------------------- §10.2 forgetting ----------
def forgetting(Ftr, Ltr, Fte, Lte, M=10000, seed=SEED):
    st = flycl(Ftr, Ltr, Fte, Lte, M=M, s=300, rho=0.3, seed=seed, keep_state=True)
    S = st["_state"]
    W, k, Wo, tc = S["W"], S["k"], S["Wo"], S["tc"]
    t0 = torch.tensor(tc[0], device=DEV)
    mte = torch.isin(Lte, t0)
    Phi0 = pwta(Fte[mte], W, k)
    fresh = st["fresh"][0]                                   # acc right after task 0
    logits = Phi0 @ Wo
    all100 = (logits.argmax(1) == Lte[mte]).float().mean().item() * 100
    # restricted to task-0's 10 classes
    restr = (t0[logits[:, t0].argmax(1)] == Lte[mte]).float().mean().item() * 100
    return {
        "task0_fresh_acc": round(fresh, 1),
        "task0_final_acc_all100": round(all100, 1),
        "task0_final_acc_restricted10": round(restr, 1),
        "measured_forgetting_task0": round(fresh - all100, 1),
        "true_catastrophic_forgetting_task0": round(fresh - restr, 1),
    }


# --------------------------------------------------- §10.3 CLS-Fly -------------
def clsfly_run(Ftr, Ltr, Fte, Lte, M, betas, seed=SEED):
    """Slow ridge readout fused with a fast KC-space class prototype (cosine).
    Returns accumulated Ā for each beta. Ridge/proto trajectories computed once."""
    torch.manual_seed(seed)
    d = Ftr.shape[1]
    W = build_W(M, d, 300, seed)
    k = int(M * 0.3)
    tc = task_split()
    Q = torch.zeros(M, NUM_CLASSES, device=DEV)
    G = torch.zeros(M, M, device=DEV)
    proto_sum = torch.zeros(NUM_CLASSES, M, device=DEV)
    seen = []
    # per-task cached scores on the accumulated test set
    per_task = {b: {t: [] for t in range(NUM_TASKS)} for b in betas}

    def zscore(x):  # standardize across class dim per sample
        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)

    for task in range(NUM_TASKS):
        cls = tc[task]
        mtr = torch.isin(Ltr, torch.tensor(cls, device=DEV))
        Phi = pwta(Ftr[mtr], W, k)
        Y = onehot(Ltr[mtr], NUM_CLASSES)
        Q += Phi.T @ Y
        G += Phi.T @ Phi
        for c in cls:
            proto_sum[c] = pwta(Ftr[Ltr == c], W, k).mean(0)
        seen += cls
        r = gcv_ridge(Phi, Y)
        Wo = chol_solve(G, Q, r)
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
    res = {}
    for b in betas:
        A = acc_metrics(per_task[b])
        res[str(b)] = round(float(np.mean(A)), 2)
    return res


# ------------------------------------------------------------------ figures ---
def make_figures(fig_dir, base, ablation, decorr, clsfly_big, clsfly_small):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ablation.png
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, key, xl in zip(ax, ["M", "rho", "s"], ["expand dim M", "coding level ρ", "synaptic degree s"]):
        xs = list(ablation[key].keys()); ys = [ablation[key][x] for x in xs]
        a.plot(range(len(xs)), ys, "o-", color="#3B6FB6")
        a.set_xticks(range(len(xs))); a.set_xticklabels(xs, rotation=0)
        a.set_xlabel(xl); a.set_ylabel("Accumulated Ā (%)"); a.grid(alpha=.3)
    fig.suptitle("Fly-CL ablations (CIFAR-100, IN21k ViT-B/16, GPU)")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/ablation.png", dpi=130); plt.close(fig)

    # decorrelation_analysis.png
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    oc = decorr["offdiag_corr"]
    bars = ["raw ViT", "dense proj", "proj + k-WTA"]
    ax[0].bar(bars, [oc["raw"], oc["dense_proj"], oc["proj_wta"]],
              color=["#9aa7b3", "#d98c5f", "#3B6FB6"])
    ax[0].set_ylabel("mean |off-diag corr|"); ax[0].set_title("Multicollinearity"); ax[0].grid(alpha=.3, axis="y")
    cs = decorr["class_similarity"]
    x = np.arange(2); w = .35
    ax[1].bar(x - w/2, [cs["raw"]["within"], cs["raw"]["between"]], w, label="raw", color="#9aa7b3")
    ax[1].bar(x + w/2, [cs["kwta"]["within"], cs["kwta"]["between"]], w, label="k-WTA", color="#3B6FB6")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["within-class", "between-class"])
    ax[1].set_ylabel("cosine similarity"); ax[1].set_title("Class geometry"); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("k-WTA decorrelation (CIFAR-100, IN21k, GPU)")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/decorrelation_analysis.png", dpi=130); plt.close(fig)

    # clsfly_extension.png
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bs = [float(b) for b in clsfly_big]; ys = [clsfly_big[b] for b in clsfly_big]
    ax[0].plot(bs, ys, "o-", color="#3B6FB6")
    ax[0].set_xlabel("fusion β"); ax[0].set_ylabel("Accumulated Ā (%)")
    ax[0].set_title("Well-determined (M=5000)"); ax[0].grid(alpha=.3)
    for Mk, series in clsfly_small.items():
        bx = [float(b) for b in series]; by = [series[b] for b in series]
        ax[1].plot(bx, by, "o-", label=f"M={Mk}")
    ax[1].set_xlabel("fusion β"); ax[1].set_ylabel("Accumulated Ā (%)")
    ax[1].set_title("Under-determined (small M)"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle("CLS-Fly: slow ridge + fast prototype (CIFAR-100, IN21k, GPU)")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/clsfly_extension.png", dpi=130); plt.close(fig)

    # accuracy_curves.png + comparison_bars.png (baselines)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in [("FlyCL", "#3B6FB6"), ("RanPAC", "#d98c5f"), ("NCM", "#9aa7b3")]:
        ax.plot(range(1, NUM_TASKS + 1), base[name]["A_t"], "o-", label=name, color=c)
    ax.set_xlabel("stage t"); ax.set_ylabel("A_t (%)"); ax.set_title("Accumulated accuracy per stage (IN21k)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/accuracy_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = ["FlyCL", "RanPAC", "NCM"]
    ax.bar(names, [base[n]["accumulated"] for n in names], color=["#3B6FB6", "#d98c5f", "#9aa7b3"])
    for i, n in enumerate(names):
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
    ap.add_argument("--cache", default="./.feat_cache/cifar100_in21k.npz",
                    help="GPU-extracted feature cache (speeds up re-runs; delete to force re-extract)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={DEV}", flush=True)

    t0 = time.time()
    Ftr, Ltr, Fte, Lte = extract_cifar_features(args.weights, args.data_root, args.cache)
    Ftr = torch.tensor(Ftr, device=DEV); Ltr = torch.tensor(Ltr, device=DEV)
    Fte = torch.tensor(Fte, device=DEV); Lte = torch.tensor(Lte, device=DEV)
    print(f"features: train {tuple(Ftr.shape)} test {tuple(Fte.shape)} ({time.time()-t0:.0f}s)", flush=True)

    # §1.2 baselines
    print("== baselines ==", flush=True)
    base = {}
    base["FlyCL"] = flycl(Ftr, Ltr, Fte, Lte, M=10000)
    print(f"FlyCL: {base['FlyCL']['accumulated']} / {base['FlyCL']['last']}", flush=True)
    base["RanPAC"] = ranpac(Ftr, Ltr, Fte, Lte, M=10000)
    print(f"RanPAC: {base['RanPAC']['accumulated']} / {base['RanPAC']['last']}", flush=True)
    base["NCM"] = ncm(Ftr, Ltr, Fte, Lte)
    print(f"NCM: {base['NCM']['accumulated']} / {base['NCM']['last']}", flush=True)
    json.dump(base, open(f"{args.out_dir}/all_methods.json", "w"), indent=1)

    # §8 ablations
    print("== ablations ==", flush=True)
    abl = ablations(Ftr, Ltr, Fte, Lte)
    json.dump(abl, open(f"{args.out_dir}/ablation_all.json", "w"), indent=2)

    # §10.1 decorrelation
    print("== decorrelation ==", flush=True)
    dec = decorrelation(Ftr, Ltr)
    print(dec["offdiag_corr"], "reduction%", dec["corr_reduction_vs_dense_pct"], flush=True)
    json.dump(dec, open(f"{args.out_dir}/decorrelation_analysis.json", "w"), indent=2)

    # §10.2 forgetting decomposition
    print("== forgetting ==", flush=True)
    frg = forgetting(Ftr, Ltr, Fte, Lte, M=10000)
    print(frg, flush=True)
    json.dump(frg, open(f"{args.out_dir}/forgetting_decomposition.json", "w"), indent=2)

    # §10.3 CLS-Fly
    print("== CLS-Fly ==", flush=True)
    betas_big = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    clsfly_big = clsfly_run(Ftr, Ltr, Fte, Lte, M=5000, betas=betas_big)
    print("clsfly M=5000:", clsfly_big, flush=True)
    clsfly_small = {}
    for M in [300, 500, 1000]:
        clsfly_small[str(M)] = clsfly_run(Ftr, Ltr, Fte, Lte, M=M, betas=[0.0, 0.1, 0.2, 0.3, 0.5])
        print(f"clsfly M={M}:", clsfly_small[str(M)], flush=True)
    json.dump(clsfly_big, open(f"{args.out_dir}/clsfly.json", "w"), indent=1)
    json.dump(clsfly_small, open(f"{args.out_dir}/clsfly_smallM.json", "w"), indent=2)

    # figures
    print("== figures ==", flush=True)
    make_figures(args.fig_dir, base, abl, dec, clsfly_big, clsfly_small)

    # rollup
    summary = {
        "_meta": {"backbone": "IN21k augreg ViT-B/16 (GPU-extracted)", "dataset": "CIFAR-100",
                  "protocol": "10 tasks, seed 1993, random split", "device": DEV},
        "baselines": {k: {"accumulated": v["accumulated"], "last": v["last"]} for k, v in base.items()},
        "ablations": abl, "decorrelation": dec, "forgetting": frg,
        "clsfly_M5000": clsfly_big, "clsfly_smallM": clsfly_small,
    }
    json.dump(summary, open(f"{args.out_dir}/level2_gpu_summary.json", "w"), indent=2)
    print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)
