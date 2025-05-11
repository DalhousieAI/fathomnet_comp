from pathlib import Path
from typing import Tuple

import open_clip                        
import torch
import torch.nn as nn
from torchvision.ops import roi_align


from utils.utils import load_model_state, _BACKBONES
from utils.models import AttentionFusion, FusedClassifier


def load_bioclip(
    pretrained: str | Path = "hf-hub:imageomics/bioclip",
    device: str | torch.device = "cuda",
    freeze: bool = True,
    precision: str = "fp32",
) -> Tuple[nn.Module, nn.Module]:
    """
    Load the BioCLIP ViT‑B/16 vision encoder.

    """

    model, _, preprocess = open_clip.create_model_and_transforms(
        pretrained,
    )

    # Keep only the vision tower so downstream code just
    # treats this like any other backbone.
    vision_encoder: nn.Module = model.visual
    vision_encoder.output_tokens = True # <<< return CLS + patch tokens
    vision_encoder.to(device).eval()

    if freeze:
        for p in vision_encoder.parameters():
            p.requires_grad = False

    # Unfreeze the last three transformer blocks of the vision encoder
    for name, param in vision_encoder.named_parameters():
        if "blocks" in name and any(block in name for block in ["9", "10", "11"]):  # Assuming blocks 9, 10, and 11 are the last three blocks
            param.requires_grad = True

    print(preprocess)
    return vision_encoder, preprocess

def load_roiencoder(
        encoder_arch: str ,
        pretrained: str | Path,
        device: str | torch.device = "cuda",
        freeze: bool = True
) -> Tuple[nn.Module, nn.Module]:

    enc = _BACKBONES[encoder_arch](weights="DEFAULT")
    roiencoder = load_model_state(
        enc, 
        pretrained, 
        origin="", 
        component="encoder", 
        custom_trained=True, 
        verbose=0
        )
    roiencoder.to(device).eval()

    if freeze:
        for p in roiencoder.parameters():
            p.requires_grad = False

    return roiencoder


# ─────────────────────────────────────────────────────────────────────────────
# Helper to enlarge each ROI by a scale factor (keeps centre, clamps to image)
# boxes  : Tensor(B, 5)  ‑>  [batch_idx, x1, y1, x2, y2]  (pixel coords)
# img_sz : Tuple[int, int] = (H, W)
# scale  : float  (e.g. 1.4 → 40 % margin)
# ─────────────────────────────────────────────────────────────────────────────
def expand_boxes(
    boxes: torch.Tensor, img_sz: Tuple[int, int], scale: float = 1.4
) -> torch.Tensor:
    _, H, W = boxes.device, *img_sz  # unpack once
    new_boxes = boxes.clone()

    # width / height + centre
    w = boxes[:, 3] - boxes[:, 1]      # x2 - x1
    h = boxes[:, 4] - boxes[:, 2]      # y2 - y1
    cx = boxes[:, 1] + 0.5 * w
    cy = boxes[:, 2] + 0.5 * h

    # scale
    w *= scale * 0.5
    h *= scale * 0.5

    new_boxes[:, 1] = (cx - w).clamp_(0, W - 1)
    new_boxes[:, 3] = (cx + w).clamp_(0, W - 1)
    new_boxes[:, 2] = (cy - h).clamp_(0, H - 1)
    new_boxes[:, 4] = (cy + h).clamp_(0, H - 1)
    return new_boxes

def extract_fused_features(context_model, fusion, imgs, boxes, expand_scale, output_size, device):
    """
    Extract fused features using ROI and context features.
    Returns:
        Fused feature vectors (B, C).
    """
    imgs, boxes = imgs.to(device), boxes.to(device)
    B, _, H, W = imgs.shape

    # Extract feature map from the context model
    with torch.no_grad():
        pooled, tokens = context_model(imgs)
        Hf, Wf = context_model.grid_size
        N_patch = Hf * Wf
        tokens = tokens[:, -N_patch:, :]
        feat_map = tokens.permute(0, 2, 1).reshape(B, tokens.size(-1), Hf, Wf)
        spatial_scale = Wf / W

    # Extract ROI features
    roi_feats = roi_align(
        feat_map, boxes,
        output_size=output_size,
        spatial_scale=spatial_scale,
        aligned=True,
    )

    # Expand boxes for context features
    ctx_boxes = expand_boxes(boxes, img_sz=(H, W), scale=expand_scale)
    ctx_feats = roi_align(
        feat_map, ctx_boxes,
        output_size=output_size,
        spatial_scale=spatial_scale,
        aligned=True,
    )

    # Mean-pool to vectors
    roi_vec = roi_feats.mean(dim=[2, 3]).to(device)
    ctx_vec = ctx_feats.mean(dim=[2, 3]).to(device)

    # Fuse ROI and context features
    fused_vec = fusion(roi_vec, ctx_vec)

    return fused_vec

def predict(
    test_loader,
    context_model,
    fusion,
    classifier,
    label_map: dict[int, str],
    device,
    output_size: int = 7,
    expand_scale: float = 1.4,
):
    """
    Run inference and return (annotation_ids, predicted_label_strings)
    """

    # reverse lookup: class_id -> label_name
    id_to_name = {v: k for k, v in label_map.items()}

    context_model.eval()
    fusion.eval()
    classifier.eval()

    annotation_ids, predictions = [], []

    with torch.no_grad():
        # expected order: imgs, boxes, ann_id
        for imgs, boxes, ann_id in test_loader:
            assert boxes.shape[-1] == 5
            assert (boxes[:,0] < imgs.size(0)).all()   #  batch_idx valid

            fused_vec = extract_fused_features(
                context_model, fusion, imgs, boxes, expand_scale, output_size, device
            )
            logits = classifier(fused_vec)

            _, pred_ids = torch.max(logits, dim=1)               # (B,)
            annotation_ids.extend(ann_id)                        # keep order
            predictions.extend(pred_ids.cpu().tolist())

    # map class‑ids → label names, default "UNK" if not found
    pred_labels = [id_to_name.get(i, "UNK") for i in predictions]

    return annotation_ids, pred_labels


def train_epoch(
    loader,
    context_model,
    fusion: AttentionFusion,
    classifier: FusedClassifier,
    criterion,
    optimizer,
    device: torch.device,
    output_size: int = 7,
    expand_scale: float = 1.4,
):

    context_model.train()
    fusion.train()
    classifier.train()

    losses, hits, total = 0.0, 0, 0

    for imgs, boxes, labels in loader:        # imgs  (B,3,H,W) ; one ROI per img

        assert boxes.shape[-1] == 5
        assert (boxes[:,0] < imgs.size(0)).all()   #  batch_idx valid
        
        labels = torch.tensor(labels)
        imgs, boxes, labels = (
            imgs.to(device),
            boxes.to(device),
            labels.to(device),
        )

        fused_vec = extract_fused_features(
            context_model, fusion, imgs, boxes, expand_scale, output_size, device
        )

        # 5  Classifier update
        logits = classifier(fused_vec)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        losses += loss.item() * imgs.size(0)
        hits += logits.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)

    return losses / total, hits / total
