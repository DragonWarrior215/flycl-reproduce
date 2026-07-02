import os
os.environ["KMP_AFFINITY"]="disabled"; os.environ["OMP_PROC_BIND"]="false"; os.environ["MKL_THREADING_LAYER"]="GNU"
import sys, torch
torch.set_num_threads(4)
sys.path.insert(0, "LibContinual")
import torch.nn as nn

class FakeBackbone(nn.Module):
    feat_dim = 768
    def forward(self, x): return torch.randn(x.shape[0], 768)

from core.model.flycl import FlyCL
kwargs = dict(init_cls_num=10, inc_cls_num=10, total_cls_num=100, task_num=10,
              embedding_dim=768, expand_dim=2000, synaptic_degree=100, coding_level=0.3,
              ridge_lower=6, ridge_upper=10, seed=1993)
m = FlyCL(FakeBackbone(), device="cpu", **kwargs)
print("W_proj shape:", tuple(m.W_proj.shape), "nnz/row:", (m.W_proj[0]!=0).sum().item(), "(expect 100)")
print("Q:", tuple(m.Q.shape), "G:", tuple(m.G.shape))
for t in range(2):
    m.before_task(t, None, None, None)
    for b in range(3):
        n=16
        data = {"image": torch.randn(n,3,224,224), "label": torch.randint(t*10,(t+1)*10,(n,))}
        pred, acc, loss = m.observe(data)
        assert loss.requires_grad, "loss must be differentiable"
    m.after_task(t, None, None, None)
    di = {"image": torch.randn(32,3,224,224), "label": torch.randint(0,(t+1)*10,(32,))}
    logits, acc = m.inference(di)
    print(f"task {t}: logits {tuple(logits.shape)} (expect [32,{(t+1)*10}]) acc={acc}")
print("Wo shape:", tuple(m.Wo.shape))
phi = m._project_wta(torch.randn(8,768))
print("WTA nonzeros/row:", (phi[0]!=0).sum().item(), f"(expect {int(2000*0.3)})")
print("SANITY PASS")
