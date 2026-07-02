"""Smoke test: confirm LibContinual's Config+Trainer path loads flycl.yaml and
builds the FlyCL classifier + vit_flycl backbone on CPU. Stops before the (slow)
full ViT training loop — construction success is what we verify here."""
import os, sys
os.environ["KMP_AFFINITY"]="disabled"; os.environ["OMP_PROC_BIND"]="false"; os.environ["MKL_THREADING_LAYER"]="GNU"
sys.dont_write_bytecode = True
os.chdir("LibContinual")
sys.path.insert(0, ".")
import torch
from core.config import Config

cfg = Config("./config/flycl.yaml").get_config_dict()
cfg["device_ids"] = "cpu"; cfg["n_gpu"] = 1
print("config loaded: dataset=%s backbone=%s classifier=%s M=%s" % (
    cfg["dataset"], cfg["backbone"]["name"], cfg["classifier"]["name"],
    cfg["classifier"]["kwargs"]["expand_dim"]))

# Build backbone + classifier exactly as Trainer._init_model does
import core.model as arch
from core.utils import get_instance
device = torch.device("cpu")
try:
    backbone = get_instance(arch, "backbone", cfg, **{"device": device})
except TypeError:
    backbone = get_instance(arch, "backbone", cfg)
model = get_instance(arch, "classifier", cfg, **{"device": device, "backbone": backbone}).to(device)
print("backbone:", type(backbone).__name__, "feat_dim=", getattr(backbone, "feat_dim", "?"))
print("classifier:", type(model).__name__)
print("W_proj:", tuple(model.W_proj.shape), "nnz/row=", int((model.W_proj[0]!=0).sum()))
print("Q:", tuple(model.Q.shape), "G:", tuple(model.G.shape))

# Exercise one before_task -> observe -> after_task -> inference cycle on random 224 images
model.before_task(0, None, None, None)
x = torch.randn(8, 3, 224, 224); y = torch.arange(8) % 10
pred, acc, loss = model.observe({"image": x, "label": y})
assert loss.requires_grad, "loss must be differentiable for Trainer.backward()"
model.after_task(0, None, None, None)
logits, a = model.inference({"image": torch.randn(8,3,224,224), "label": torch.arange(8)%10})
print("observe/after_task/inference cycle OK; logits", tuple(logits.shape))
print("SMOKE PASS: framework builds and runs FlyCL end-to-end on CPU")
