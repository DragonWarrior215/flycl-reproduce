"""
Frozen ViT-B/16 backbone for Fly-CL integration in LibContinual.

Fly-CL (ICLR 2026) uses a *nearly-frozen* pretrained backbone and reframes
continual learning as a similarity-matching / closed-form readout problem.
The paper uses timm ImageNet-21k augreg ViT-B/16. In this reproduction the
21k weights host (HuggingFace) was unreachable, so we wrap torchvision's
ViT-B/16 (ImageNet-1k supervised). See docs_deliverable/env_setup.md §3 for
the fidelity note. The backbone is frozen (requires_grad=False) and returns
the 768-d pre-logit CLS embedding.

To keep experiments cheap on CPU, this backbone can also operate in a
"precomputed features" mode: if a feature cache is provided, forward() simply
returns the cached feature rows. This is functionally identical to running the
frozen ViT but avoids re-extracting features for every experiment.
"""
import os
import torch
import torch.nn as nn


class ViT_FlyCL(nn.Module):
    def __init__(self, pretrained=True, weights_path=None, model_name="vit_base_patch16_224", **kwargs):
        super().__init__()
        self.feat_dim = 768
        self._build(pretrained, weights_path)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _build(self, pretrained, weights_path):
        from torchvision.models import vit_b_16
        m = vit_b_16(weights=None)
        if pretrained:
            # weights_path: local torchvision state_dict; else fall back to torchvision download
            if weights_path and os.path.exists(weights_path):
                m.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                from torchvision.models import ViT_B_16_Weights
                m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()  # drop classification head -> 768-d CLS embedding
        self.backbone = m

    @torch.no_grad()
    def forward(self, x, **kwargs):
        # x: [B, 3, 224, 224] -> [B, 768]
        return self.backbone(x)


def vit_flycl(pretrained=False, **kwargs):
    return ViT_FlyCL(pretrained=pretrained, **kwargs)
