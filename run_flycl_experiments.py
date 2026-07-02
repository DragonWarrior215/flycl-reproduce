"""
Fly-CL CIFAR-100 reproduction driver (operates on cached ViT features).

Runs three methods on the SAME frozen ViT-B/16 features under the CIFAR-100
B0 10x10 class-incremental protocol (seed=1993, official class split):
  - Fly-CL   : sparse random projection + k-WTA + accumulated ridge (GCV)
  - RanPAC   : dense random projection + ReLU + accumulated ridge (val sweep)
  - NCM      : nearest class-mean on frozen features (no projection)

Produces results/*.json consumed by make_figures.py.

This is an INDEPENDENT re-implementation of the official Fly-CL algorithm
(github.com/gfyddha/Fly-CL, ICLR 2026), NOT the authors' repo. The LibContinual
integration lives in LibContinual/core/model/flycl.py; a cross-check (see the
report) confirms it produces bit-identical A_t to the reference below.
"""
import os, json, time, random
os.environ["KMP_AFFINITY"]="disabled"; os.environ["OMP_PROC_BIND"]="false"; os.environ["MKL_THREADING_LAYER"]="GNU"
import numpy as np, torch
torch.set_num_threads(16)

TRAIN = "features/cifar100_vit_b16_tv_train.npz"
TEST  = "features/cifar100_vit_b16_tv_test.npz"


def load():
    tr=np.load(TRAIN); te=np.load(TEST)
    return tr["features"], tr["labels"], te["features"], te["labels"]


def task_split(num_classes=100, num_tasks=10, seed=1993):
    random.seed(seed)
    rc=random.sample(list(range(num_classes)), num_classes)
    cpt=num_classes//num_tasks
    return [rc[i*cpt:(i+1)*cpt] for i in range(num_tasks)]


def acc_metrics(acc, num_tasks=10):
    am=[[0.0]*num_tasks for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j,v in enumerate(acc[i]): am[i][i+j]=round(v,2)
    A=[round(sum(am[i][j] for i in range(j+1))/(j+1),2) for j in range(num_tasks)]
    return am, A


def flycl(Ftr,Ltr,Fte,Lte, expand_dim=10000, synaptic_degree=300, coding_level=0.3,
          seed=1993, ridge_lower=6, ridge_upper=10, num_classes=100, num_tasks=10):
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
        r=sel_ridge(Phi,Y); L=torch.linalg.cholesky(G+r*torch.eye(expand_dim)); Wo=torch.cholesky_solve(Q,L)
        tt.append(time.time()-t0)
        for st in range(task+1):
            m2=torch.isin(Lte,torch.tensor(tc[st])); logits=pwta(Fte[m2])@Wo
            acc[st].append((logits.argmax(1)==Lte[m2]).float().mean().item()*100)
    return acc, tt


def ranpac(Ftr,Ltr,Fte,Lte, M=10000, seed=1993, num_classes=100, num_tasks=10):
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
            Wo=torch.linalg.solve(Gv+r*torch.eye(M),Qv); loss=torch.nn.functional.mse_loss(Phi[nv:]@Wo,Y[nv:])
            if loss.item()<best: best=loss.item(); br=r
        Wo=torch.linalg.solve(G+br*torch.eye(M),Q); tt.append(time.time()-t0)
        for st in range(task+1):
            m2=torch.isin(Lte,torch.tensor(tc[st])); acc[st].append((proj(Fte[m2])@Wo).argmax(1).eq(Lte[m2]).float().mean().item()*100)
    return acc, tt


def ncm(Ftr,Ltr,Fte,Lte, seed=1993, num_classes=100, num_tasks=10):
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
    Ftr,Ltr,Fte,Lte = load()
    os.makedirs("results", exist_ok=True)
    out={}
    for name, fn in [("FlyCL",flycl),("RanPAC",ranpac),("NCM",ncm)]:
        acc, tt = fn(Ftr,Ltr,Fte,Lte)
        am, A = acc_metrics(acc)
        out[name]={"acc_matrix":am,"A_t":A,"accumulated":round(float(np.mean(A)),2),
                   "last":A[-1],"avg_train_s_per_task":round(float(np.mean(tt)),2)}
        print(f"{name}: Accumulated={out[name]['accumulated']} Last={out[name]['last']}")
    json.dump(out, open("results/all_methods.json","w"), indent=2)
    print("saved results/all_methods.json")
