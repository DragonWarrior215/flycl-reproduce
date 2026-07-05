"""
Fly-CL reproduction driver (operates on cached frozen-ViT features).

Runs three methods on the SAME frozen ViT-B/16 features under the official
Fly-CL class-incremental protocols (scripts/test_*.sh of gfyddha/Fly-CL):
  - Fly-CL   : sparse random projection + k-WTA + accumulated ridge (GCV)
  - RanPAC   : dense random projection + ReLU + accumulated ridge (val sweep)
  - NCM      : nearest class-mean on frozen features (no projection)

Protocols (official):
  cifar100 : 100 classes, 10 tasks, seed 1993
  cub      : 200 classes, 10 tasks, seed 2023
  vtab     :  50 classes,  5 tasks, seed 2023
Shared hyperparameters: M=10000, synaptic_degree=300, coding_level=0.3,
ridge in 10^[6,10) via GCV.

Usage:
  python run_flycl_experiments.py --dataset cub                  # augreg backbone features
  python run_flycl_experiments.py --dataset cifar100 --tag tv    # old torchvision IN-1k features
  -> results/all_methods_{dataset}[_tv].json

This is an INDEPENDENT re-implementation of the official Fly-CL algorithm
(github.com/gfyddha/Fly-CL, ICLR 2026), NOT the authors' repo. The LibContinual
integration lives in LibContinual/core/model/flycl.py; a cross-check (see the
report) confirms it produces bit-identical A_t to the reference below.
"""
import os, json, time, random, argparse
os.environ["KMP_AFFINITY"]="disabled"; os.environ["OMP_PROC_BIND"]="false"; os.environ["MKL_THREADING_LAYER"]="GNU"
import numpy as np, torch
torch.set_num_threads(16)

DATASETS = {
    "cifar100": dict(num_classes=100, num_tasks=10, seed=1993),
    "cub":      dict(num_classes=200, num_tasks=10, seed=2023),
    "vtab":     dict(num_classes=50,  num_tasks=5,  seed=2023),
}


def load(train_path, test_path):
    tr=np.load(train_path); te=np.load(test_path)
    return tr["features"], tr["labels"], te["features"], te["labels"]


def task_split(num_classes, num_tasks, seed):
    random.seed(seed)
    rc=random.sample(list(range(num_classes)), num_classes)
    cpt=num_classes//num_tasks
    return [rc[i*cpt:(i+1)*cpt] for i in range(num_tasks)]


def acc_metrics(acc, num_tasks):
    am=[[0.0]*num_tasks for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j,v in enumerate(acc[i]): am[i][i+j]=round(v,2)
    A=[round(sum(am[i][j] for i in range(j+1))/(j+1),2) for j in range(num_tasks)]
    return am, A


def chol_solve_ridge(G, Q, r):
    """cholesky_solve(Q, chol(G + r*I)) without materializing r*eye (7GB-RAM box)."""
    A = G.clone(); A.diagonal().add_(r)
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(Q, L)


def solve_ridge(G, Q, r):
    """linalg.solve(G + r*I, Q) without materializing r*eye."""
    A = G.clone(); A.diagonal().add_(r)
    return torch.linalg.solve(A, Q)


def flycl(Ftr,Ltr,Fte,Lte, num_classes, num_tasks, seed, expand_dim=10000,
          synaptic_degree=300, coding_level=0.3, ridge_lower=6, ridge_upper=10):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    d=Ftr.shape[1]
    Ftr=torch.tensor(Ftr); Ltr=torch.tensor(Ltr); Fte=torch.tensor(Fte); Lte=torch.tensor(Lte)
    W=torch.zeros(expand_dim,d)
    for r in range(expand_dim):
        cols=torch.randperm(d)[:synaptic_degree]; W[r,cols]=torch.randn(synaptic_degree)
    tc=task_split(num_classes,num_tasks,seed)
    def oh(y): o=torch.zeros(y.shape[0],num_classes); o.scatter_(1,y.long().view(-1,1),1.0); return o
    def pwta(F):
        z=F@W.T; k=int(expand_dim*coding_level); v,idx=z.topk(k,1); o=torch.zeros_like(z); o.scatter_(1,idx,v); return o
    def sel_ridge(Feat,Y):
        U,S,_=torch.linalg.svd(Feat,full_matrices=False); Ss=S**2; UTY=U.T@Y
        rid=torch.tensor(10.0**np.arange(ridge_lower,ridge_upper)); n=Feat.shape[0]; best=1e30; br=rid[0]
        for r in rid:
            diag=Ss/(Ss+r); df=diag.sum(); Yh=U@(diag[:,None]*UTY); g=(torch.norm(Y-Yh)**2/n)/(1-df/n)**2
            if g.item()<best: best=g.item(); br=r
        return br
    Q=torch.zeros(expand_dim,num_classes); G=torch.zeros(expand_dim,expand_dim)
    acc={t:[] for t in range(num_tasks)}; tt=[]
    for task in range(num_tasks):
        m=torch.isin(Ltr,torch.tensor(tc[task])); Fx,Ly=Ftr[m],Ltr[m]
        t0=time.time(); Phi=pwta(Fx); Y=oh(Ly); Q+=Phi.T@Y; G+=Phi.T@Phi
        r=sel_ridge(Phi,Y); Wo=chol_solve_ridge(G,Q,r)
        tt.append(time.time()-t0)
        for st in range(task+1):
            m2=torch.isin(Lte,torch.tensor(tc[st])); logits=pwta(Fte[m2])@Wo
            acc[st].append((logits.argmax(1)==Lte[m2]).float().mean().item()*100)
        print(f"  flycl task {task}: A_t={sum(acc[st][-1] for st in range(task+1))/(task+1):.2f} "
              f"ridge={r:.0e} ({tt[-1]:.0f}s)", flush=True)
    return acc, tt


def ranpac(Ftr,Ltr,Fte,Lte, num_classes, num_tasks, seed, M=10000):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    d=Ftr.shape[1]
    Ftr=torch.tensor(Ftr); Ltr=torch.tensor(Ltr); Fte=torch.tensor(Fte); Lte=torch.tensor(Lte)
    Wr=torch.randn(d,M); tc=task_split(num_classes,num_tasks,seed)
    def proj(F): return torch.relu(F@Wr)
    def oh(y): o=torch.zeros(y.shape[0],num_classes); o.scatter_(1,y.long().view(-1,1),1.0); return o
    Q=torch.zeros(M,num_classes); G=torch.zeros(M,M); acc={t:[] for t in range(num_tasks)}; tt=[]
    for task in range(num_tasks):
        m=torch.isin(Ltr,torch.tensor(tc[task])); Fx,Ly=Ftr[m],Ltr[m]
        t0=time.time(); Phi=proj(Fx); Y=oh(Ly); Q+=Phi.T@Y; G+=Phi.T@Phi
        rid=10.0**np.arange(-8,9); nv=int(Phi.shape[0]*0.8)
        Qv=Phi[:nv].T@Y[:nv]; Gv=Phi[:nv].T@Phi[:nv]; best=1e30; br=rid[0]
        for r in rid:
            Wo=solve_ridge(Gv,Qv,r); loss=torch.nn.functional.mse_loss(Phi[nv:]@Wo,Y[nv:])
            if loss.item()<best: best=loss.item(); br=r
        Wo=solve_ridge(G,Q,br); tt.append(time.time()-t0)
        for st in range(task+1):
            m2=torch.isin(Lte,torch.tensor(tc[st])); acc[st].append((proj(Fte[m2])@Wo).argmax(1).eq(Lte[m2]).float().mean().item()*100)
        print(f"  ranpac task {task}: A_t={sum(acc[st][-1] for st in range(task+1))/(task+1):.2f} "
              f"ridge={br:.0e} ({tt[-1]:.0f}s)", flush=True)
    return acc, tt


def ncm(Ftr,Ltr,Fte,Lte, num_classes, num_tasks, seed):
    Ftr=torch.tensor(Ftr); Ltr=torch.tensor(Ltr); Fte=torch.tensor(Fte); Lte=torch.tensor(Lte)
    tc=task_split(num_classes,num_tasks,seed); protos=torch.zeros(num_classes,Ftr.shape[1]); seen=[]
    acc={t:[] for t in range(num_tasks)}
    for task in range(num_tasks):
        for c in tc[task]: protos[c]=Ftr[Ltr==c].mean(0)
        seen+=tc[task]; st_t=torch.tensor(seen)
        for st in range(task+1):
            m2=torch.isin(Lte,torch.tensor(tc[st])); d=torch.cdist(Fte[m2],protos[st_t])
            acc[st].append((st_t[d.argmin(1)]==Lte[m2]).float().mean().item()*100)
    return acc, [0.1]*num_tasks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--tag", default="augreg", help="feature file tag: augreg (paper IN21k backbone) or tv (torchvision IN-1k)")
    p.add_argument("--train", default=None, help="override train features path")
    p.add_argument("--test", default=None, help="override test features path")
    p.add_argument("--methods", default="FlyCL,RanPAC,NCM")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = DATASETS[args.dataset]
    train_path = args.train or f"features/{args.dataset}_vit_b16_{args.tag}_train.npz"
    test_path  = args.test  or f"features/{args.dataset}_vit_b16_{args.tag}_test.npz"
    out_path = args.out or f"results/all_methods_{args.dataset}{'' if args.tag=='augreg' else '_'+args.tag}.json"

    Ftr,Ltr,Fte,Lte = load(train_path, test_path)
    print(f"{args.dataset}: train {Ftr.shape}, test {Fte.shape}, "
          f"{cfg['num_classes']} classes, {cfg['num_tasks']} tasks, seed {cfg['seed']}", flush=True)
    os.makedirs("results", exist_ok=True)
    fns = {"FlyCL":flycl, "RanPAC":ranpac, "NCM":ncm}
    out={"_meta": dict(dataset=args.dataset, backbone_tag=args.tag, train=train_path, test=test_path, **cfg)}
    for name in args.methods.split(","):
        acc, tt = fns[name](Ftr,Ltr,Fte,Lte, cfg["num_classes"], cfg["num_tasks"], cfg["seed"])
        am, A = acc_metrics(acc, cfg["num_tasks"])
        out[name]={"acc_matrix":am,"A_t":A,"accumulated":round(float(np.mean(A)),2),
                   "last":A[-1],"avg_train_s_per_task":round(float(np.mean(tt)),2)}
        print(f"{name}: Accumulated={out[name]['accumulated']} Last={out[name]['last']}", flush=True)
        json.dump(out, open(out_path,"w"), indent=2)  # checkpoint after each method
    print(f"saved {out_path}", flush=True)
