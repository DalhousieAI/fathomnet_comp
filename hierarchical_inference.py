#!/usr/bin/env python
# hierarchical_meta_fusion.py
# -----------------------------------------------------------------
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from utils.dataset import FathomNetDataset
from utils.utils    import build_model, map_label_to_idx, convert_to_rgb, save_predictions_to_csv

# ----------------------- paths & cfg -----------------------------
CKPT_DIRS = {                                           # ← fill real paths
    "Phylum":  "models/vit_l_16_pre-None_cls-one_hot_rank-Phylum_seed-42_e-100_aug-True_isz-224",
    "Class":   "models/vit_l_16_pre-model_cls-one_hot_rank-Class_seed-42_e-20_aug-True_isz-224",
    "Order":   "models/vit_l_16_pre-model_cls-one_hot_rank-Order_seed-42_e-20_aug-True_isz-224",
    "Family":  "models/vit_l_16_pre-model_cls-one_hot_rank-Family_seed-42_e-20_aug-True_isz-224",
    "Genus":   "models/vit_l_16_pre-model_cls-one_hot_rank-Genus_seed-42_e-20_aug-True_isz-224",
    "Species": "models/vit_l_16_pre-model_cls-one_hot_rank-Species_seed-42_e-20_aug-True_isz-224",
}
TRAIN_CSV      = Path("cfg/hierarchy/hierarchy_labels_train_noNone.csv")
TEST_CSV       = Path("../data/test/annotations.csv")
META_WEIGHTS   = Path("meta_classifier.pth")
OUT_CSV        = Path("species_predictions.csv")
BATCH_SIZE     = 32
HIDDEN_DIM     = 256                                    # meta‑MLP width
EPOCHS_META    = 8
LR_META        = 1e-3
# -----------------------------------------------------------------


# --------------------- meta‑classifier ---------------------------
class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------- utilities ---------------------------------
def build_dataloader(df: pd.DataFrame, transforms, labelled=True):
    return torch.utils.data.DataLoader(
        FathomNetDataset(df, label_col="Species", transform=transforms, is_test=not labelled),
        batch_size=BATCH_SIZE, shuffle=labelled,
        num_workers=4, pin_memory=True
    )


def collect_rank_logits(loader, rank_models, device, want_ids=False):
    """
    Returns:
        rank_logits : dict  rank -> tensor(N, C_r)
        labels      : list[int]   (empty if loader is test)
        ids         : list[str]   (only if want_ids=True)
    """
    N = len(loader.dataset)
    rank_logits = {r: torch.zeros(N, m["out_dim"], device=device)
                   for r, m in rank_models.items()}

    labels, ids = [], []
    pos = 0
    with torch.no_grad():
        for batch in loader:
            if want_ids:
                imgs, batch_ids = batch
            else:                       # labelled train loader
                imgs, batch_lbl = batch
                labels.extend(batch_lbl)

            bs = imgs.size(0)
            imgs = imgs.to(device)
            for r, m in rank_models.items():
                rank_logits[r][pos:pos + bs] = m["model"](imgs)
            pos += bs

            if want_ids:
                ids.extend(batch_ids)

    return rank_logits, labels, ids



# --------------------- main --------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Lambda(convert_to_rgb),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # -------- label maps -----------
    train_df = pd.read_csv(TRAIN_CSV)
    rank2lbl = {r: map_label_to_idx(train_df, r)[1] for r in CKPT_DIRS}
    n_species = len(rank2lbl["Species"])
    idx2sp = {v: k for k, v in rank2lbl["Species"].items()}

    # -------- build & load rank models once -----------
    rank_models = {}
    for rank, ckpt_dir in CKPT_DIRS.items():
        out_dim = len(rank2lbl[rank])
        m = build_model("vit_l_16", "one_hot", None, requires_grad=False, output_dim=out_dim).to(device)
        m.load_state_dict(torch.load(Path(ckpt_dir)/"model.ckpt", map_location=device))
        m.eval()
        rank_models[rank] = {"model": m, "out_dim": out_dim}

    concat_dim = sum(m["out_dim"] for m in rank_models.values())

    # -------- train meta‑MLP if weights not found -----------
    if not META_WEIGHTS.exists():
        print("Training meta‑classifier …")
        loader_train = build_dataloader(train_df, tfm, labelled=True)
        rank_logits, str_labels, _ = collect_rank_logits(loader_train, rank_models, device, want_ids=False)

        feats = torch.cat([rank_logits[r] for r in CKPT_DIRS], dim=1)
        int_labels = [rank2lbl["Species"][name] for name in str_labels]
        labels = torch.tensor(int_labels, device=device)

        meta = MetaMLP(concat_dim, n_species).to(device)
        opt  = torch.optim.AdamW(meta.parameters(), lr=LR_META)
        for epoch in range(EPOCHS_META):
            idx = torch.randperm(feats.size(0), device=device)
            for i in range(0, feats.size(0), BATCH_SIZE):
                batch = idx[i:i+BATCH_SIZE]
                logits = meta(feats[batch])
                loss = F.cross_entropy(logits, labels[batch])
                opt.zero_grad(); loss.backward(); opt.step()
            print(f"epoch {epoch+1}/{EPOCHS_META}  loss {loss.item():.4f}")

        torch.save(meta.state_dict(), META_WEIGHTS)
        print(f"Saved meta weights → {META_WEIGHTS}")
    else:
        meta = MetaMLP(concat_dim, n_species).to(device)
        meta.load_state_dict(torch.load(META_WEIGHTS, map_location=device))
        meta.eval()
        print("Loaded meta‑classifier weights.")

    # -------- inference on test set -----------
    test_df  = pd.read_csv(TEST_CSV)
    loader_test = build_dataloader(test_df, tfm, labelled=False)
    rank_logits_test, _, ann_ids = collect_rank_logits(loader_test, rank_models, device, want_ids=True)
    feats_test = torch.cat([rank_logits_test[r].softmax(1) for r in CKPT_DIRS], dim=1)

    with torch.no_grad():
        preds_idx = meta(feats_test).argmax(1).cpu().tolist()
    preds = [idx2sp.get(i, "UNK") for i in preds_idx]   # guard against >79

    # -------- write csv -----------
    save_predictions_to_csv(ann_ids, preds, OUT_CSV)
    print(f"Wrote predictions to {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
