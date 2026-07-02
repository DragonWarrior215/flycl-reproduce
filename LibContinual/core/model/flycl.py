"""
Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and
Reduced Training Time in Pre-trained Model-based Continual Representation Learning.

ICLR 2026.  arXiv:2510.16877.  Official code: https://github.com/gfyddha/Fly-CL

This is an independent re-implementation refactored into the LibContinual model
contract (observe / inference / before_task / after_task / get_parameters).
It is NOT the authors' repo. Algorithm faithfully follows the official main.py:

  frozen ViT feature x in R^d
   -> sparse random projection  z = W x,  W in R^{M x d}, each row has `s`
      non-zero N(0,1) weights  (PN -> KC, fly olfactory expansion)
   -> k-WTA sparsification: keep top ceil(coding_level * M) activations, zero
      the rest  (APL lateral inhibition)
   -> accumulate  Q += Phi^T Y,  G += Phi^T Phi  across tasks
   -> ridge readout  Wo = (G + lambda I)^{-1} Q   via Cholesky, lambda by GCV
   -> predict argmax(Phi(x) Wo)

Because G and Q are sufficient statistics over ALL data seen, the closed-form
solution equals ridge regression over the union of tasks -> structurally
forgetting-free (order-independent).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlyCL(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.device = device

        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.total_cls_num = kwargs["total_cls_num"]
        self.task_num = kwargs["task_num"]

        # Fly-CL hyperparameters (defaults = official test_cifar.sh)
        self.embedding_dim = kwargs.get("embedding_dim", getattr(backbone, "feat_dim", 768))
        self.expand_dim = kwargs.get("expand_dim", 10000)          # M
        self.synaptic_degree = kwargs.get("synaptic_degree", 300)  # s non-zeros per KC row
        self.coding_level = kwargs.get("coding_level", 0.3)        # WTA keep-ratio rho
        self.ridge_lower = kwargs.get("ridge_lower", 6)            # log10 lower
        self.ridge_upper = kwargs.get("ridge_upper", 10)           # log10 upper
        self.seed = kwargs.get("seed", 1993)

        self._known_classes = 0
        self._classes_seen_so_far = 0

        # Build the sparse fly projection matrix W in R^{M x d}
        self._build_projection()

        # Accumulated sufficient statistics (grown lazily to total classes)
        self.Q = torch.zeros(self.expand_dim, self.total_cls_num, device=self.device)
        self.G = torch.zeros(self.expand_dim, self.expand_dim, device=self.device)
        # Readout weight Wo in R^{M x C}
        self.Wo = torch.zeros(self.expand_dim, self.total_cls_num, device=self.device)

        self.backbone.to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    # ---- fly projection (PN -> KC): sparse random weights ------------------
    def _build_projection(self):
        g = torch.Generator().manual_seed(self.seed)
        M, d, s = self.expand_dim, self.embedding_dim, self.synaptic_degree
        W = torch.zeros(M, d)
        for row in range(M):
            cols = torch.randperm(d, generator=g)[:s]
            W[row, cols] = torch.randn(s, generator=g)
        # store as sparse for memory/speed (mm with dense feature matrix)
        self.register_buffer("W_proj", W)  # [M, d]

    # ---- k-WTA sparsification (APL lateral inhibition) ---------------------
    def _project_wta(self, feats):
        """feats: [N, d] -> Phi: [N, M] sparse code (k-WTA)."""
        z = feats.to(self.device) @ self.W_proj.T.to(self.device)  # [N, M]
        k = max(1, int(self.expand_dim * self.coding_level))
        vals, idx = z.topk(k, dim=1, largest=True)
        phi = torch.zeros_like(z)
        phi.scatter_(1, idx, vals)
        return phi

    # ---- GCV ridge selection (official select_ridge_parameter) -------------
    def _select_ridge(self, Phi, Y):
        X = Phi
        U, S, _ = torch.linalg.svd(X, full_matrices=False)
        S_sq = S ** 2
        UTY = U.T @ Y
        ridges = torch.tensor(10.0 ** np.arange(self.ridge_lower, self.ridge_upper), device=X.device)
        n = X.shape[0]
        best, best_ridge = float("inf"), ridges[0]
        for ridge in ridges:
            diag = S_sq / (S_sq + ridge)
            df = diag.sum()
            Y_hat = U @ (diag[:, None] * UTY)
            resid = torch.norm(Y - Y_hat) ** 2
            gcv = (resid / n) / (1 - df / n) ** 2
            if gcv.item() < best:
                best, best_ridge = gcv.item(), ridge
        return best_ridge

    # ---- LibContinual model contract ---------------------------------------
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        if task_idx == 0:
            self._classes_seen_so_far = self.init_cls_num
        else:
            self._classes_seen_so_far += self.inc_cls_num
        # accumulate this task's features for the closed-form solve at after_task
        self._task_feats = []
        self._task_labels = []

    def observe(self, data):
        """No backprop. Extract frozen features and stash for after_task solve.
        Return a dummy differentiable zero loss so Trainer's loss.backward() is a no-op."""
        x = data["image"].to(self.device)
        y = data["label"]
        with torch.no_grad():
            feats = self.backbone(x).cpu()
        self._task_feats.append(feats)
        self._task_labels.append(y.cpu())
        dummy = torch.zeros(1, device=self.device, requires_grad=True)
        return None, 0.0, dummy

    @torch.no_grad()
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        self._known_classes = self._classes_seen_so_far
        feats = torch.cat(self._task_feats, dim=0)           # [N, d]
        labels = torch.cat(self._task_labels, dim=0)         # [N]
        Phi = self._project_wta(feats)                       # [N, M]
        Y = F.one_hot(labels.to(self.device), self.total_cls_num).float()  # [N, C_total]

        self.Q += Phi.T @ Y
        self.G += Phi.T @ Phi

        # GCV lambda selected on current-task sparse code vs its one-hot targets
        ridge = self._select_ridge(Phi, Y)
        L = torch.linalg.cholesky(self.G + ridge * torch.eye(self.expand_dim, device=self.device))
        self.Wo = torch.cholesky_solve(self.Q, L)            # [M, C_total]
        print(f"[Fly-CL] task {task_idx}: N={feats.shape[0]} classes_seen={self._classes_seen_so_far} lambda={ridge.item():.1e}")
        del self._task_feats, self._task_labels

    @torch.no_grad()
    def inference(self, data):
        x = data["image"].to(self.device)
        y = data["label"]
        feats = self.backbone(x).cpu()
        Phi = self._project_wta(feats)                       # [B, M]
        # Score against ALL columns (official main.py convention). Untrained classes
        # have ~zero readout weight (their Q columns are still zero), so they never win.
        # NB: class order may be shuffled, so restricting to the first N columns would be
        # wrong; the full-Wo argmax is both faithful and correct for class-incremental eval.
        logits = Phi @ self.Wo                               # [B, total_cls_num]
        preds = logits.argmax(dim=1).cpu()
        correct = preds.eq(y.expand_as(preds)).sum().item()
        acc = round(correct / len(y), 4)
        return logits, acc

    def get_parameters(self, config):
        # nothing to optimize (frozen backbone + closed-form head); return dummy
        return [p for p in self.backbone.parameters()]
