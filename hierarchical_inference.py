#!/usr/bin/env python
# hierarchical_meta_fusion_attention.py
# ---------------------------------------------------------------
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from utils.dataset import FathomNetDataset
from utils.utils    import build_model, map_label_to_idx, convert_to_rgb, save_predictions_to_csv
# ---------------------------------------------------------------

# ---------------- CONFIG --------------------------------------
CKPT_DIRS = {                                           
    "Phylum":  "models/wide_resnet101_2_pre-None_cls-one_hot_rank-Phylum_seed-2048_e-60_aug-True_isz-112lr-0.0001_n-heads-100",
    "Class":   "models/wide_resnet101_2_pre-model_cls-one_hot_rank-Class_seed-2048_e-30_aug-True_isz-112lr-0.0001_n-heads-100",
    "Order":   "models/wide_resnet101_2_pre-model_cls-one_hot_rank-Order_seed-2048_e-30_aug-True_isz-112lr-0.0001_n-heads-100",
    "Family":  "models/wide_resnet101_2_pre-model_cls-one_hot_rank-Family_seed-2048_e-30_aug-True_isz-112lr-0.0001_n-heads-100",
    "Genus":   "models/wide_resnet101_2_pre-model_cls-one_hot_rank-Genus_seed-2048_e-30_aug-True_isz-112lr-0.0001_n-heads-100",
    "Species": "models/wide_resnet101_2_pre-model_cls-one_hot_rank-Species_seed-2048_e-30_aug-True_isz-112lr-0.0001_n-heads-100",
}
TRAIN_CSV      = Path("cfg/hierarchy/hierarchy_labels_train_noNone.csv")
TEST_CSV       = Path("../data/test/annotations.csv")
META_WEIGHTS   = Path("meta_classifier_attention.pth")
OUT_CSV        = Path("species_predictions.csv")

BATCH_SIZE     = 32
HIDDEN_DIM  = 128
N_ATT_HEADS = 2
EPOCHS_META    = 5
LR_META        = 5e-4
LAMBDA_CONS    = 0.01         # weight for hierarchy consistency penalty
PENALTY_WARMUP_EPOCHS = 5
# ---------------------------------------------------------------


# ============ META FUSION MODEL WITH ATTENTION =================
RANK_ORDER = list(CKPT_DIRS.keys())[:-1]    # exclude Species → only parents

class MetaFusion(nn.Module):
    """
    1. project each rank's feature vector to a common d_model
    2. apply one MH self-attention layer
    3. average the attended sequence
    4. final classifier -> species logits
    """
    def __init__(self, rank_dims: Dict[str, int], n_species: int):
        super().__init__()
        self.proj = nn.ModuleDict({r: nn.Sequential(
                               nn.Linear(dim, HIDDEN_DIM),
                               nn.LayerNorm(HIDDEN_DIM))
                           for r, dim in rank_dims.items()})

        self.attn = nn.MultiheadAttention(HIDDEN_DIM, N_ATT_HEADS,
                                          batch_first=True)
        self.classifier = nn.Linear(HIDDEN_DIM, n_species)
        self.dropout = nn.Dropout(0.2)

        

    def forward(self, feats_per_rank: Dict[str, torch.Tensor]):   # B × C_r
        # keep ranks in fixed order for stacking
        seq = [self.proj[r](feats_per_rank[r]) for r in RANK_ORDER]
        x   = torch.stack(seq, dim=1)            # (B, R, d_model)
        x,_ = self.attn(x,x,x)
        x   = self.dropout(x).mean(dim=1)
        
        return self.classifier(x)                # (B, n_species)
# ===============================================================


# --------------------- utilities (unchanged) -------------------
def build_dataloader(df: pd.DataFrame, transforms, labelled=True):
    return torch.utils.data.DataLoader(
        FathomNetDataset(df, label_col="Species", transform=transforms,
                         is_test=not labelled),
        batch_size=BATCH_SIZE, shuffle=labelled,
        num_workers=0, pin_memory=True
    )

def _split_heads(raw_logits, out_dim, num_heads):
    if isinstance(raw_logits, (list, tuple)):
        assert len(raw_logits) == num_heads
        return raw_logits
    if raw_logits.dim() == 3:          # (B,H,C) or (H,B,C)
        if raw_logits.size(1) == num_heads:
            return [raw_logits[:,h,:] for h in range(num_heads)]
        if raw_logits.size(0) == num_heads:
            return [raw_logits[h,:,:] for h in range(num_heads)]
    # flattened (B, H*C)
    B = raw_logits.size(0)
    return raw_logits.view(B, num_heads, out_dim).unbind(1)

def collect_rank_logits(loader, rank_models, device, num_heads, want_ids=False):
    N = len(loader.dataset)
    rank_logits = {r: [torch.zeros(N, m["out_dim"], device=device)
                       for _ in range(num_heads)]
                   for r, m in rank_models.items()}

    labels, ids, pos = [], [], 0
    with torch.no_grad():
        for batch in loader:
            imgs, xtra = batch
            if want_ids: ids.extend(xtra)
            else:        labels.extend(xtra)

            bs = imgs.size(0)
            imgs = imgs.to(device)

            for r, m in rank_models.items():
                outs = _split_heads(m["model"](imgs),
                                    m["out_dim"], num_heads)
                for h in range(num_heads):
                    rank_logits[r][h][pos:pos+bs] = outs[h]
            pos += bs
    return rank_logits, labels, ids
# ---------------------------------------------------------------


# ------------------------- main ---------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_arch, num_heads, img_size = "wide_resnet101_2", 100, (112,112)

    tfm = T.Compose([
        T.Resize(img_size), T.ToTensor(), T.Lambda(convert_to_rgb),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # ------ labels & hierarchy maps -----------------------------
    train_df = pd.read_csv(TRAIN_CSV)
    rank2lbl = {r: map_label_to_idx(train_df, r)[1] for r in CKPT_DIRS}
    n_species = len(rank2lbl["Species"])
    idx2sp    = {v:k for k,v in rank2lbl["Species"].items()}

    # ------ hierarchy helpers -------------------------------------
    # 1) parent_maps      : species_idx -> parent_idx  (Python dict of dicts)
    # 2) parent_targets   : rank -> LongTensor(N_train)
    # 3) mapping_mats     : rank -> Tensor(n_species , |rank|)

    parent_maps: Dict[str, Dict[int, int]] = {r: {} for r in RANK_ORDER}

    for _, row in train_df.iterrows():
        sp_label = row["Species"]
        if pd.isna(sp_label):
            continue
        sp_idx = rank2lbl["Species"][sp_label]

        for rank in RANK_ORDER:
            par_label = row[rank]
            if pd.isna(par_label):
                continue
            parent_maps[rank][sp_idx] = rank2lbl[rank][par_label]

    # ---------- parent_targets (sample-wise ground-truth parents) ----------
    parent_targets = {
        rank: torch.tensor(
            [rank2lbl[rank][lab]                       # safe: rank2lbl handles str → idx
            if not pd.isna(lab)
            else 0                                    # fallback idx 0 for missing label
            for lab in train_df[rank]],
            device=device,
            dtype=torch.long,
        )
        for rank in RANK_ORDER
    }

    # ---------- mapping matrices (differentiable aggregation) -------------
    mapping_mats = {}
    for rank in RANK_ORDER:
        n_parents = len(rank2lbl[rank])
        if parent_maps[rank]:
            rows, cols = zip(*parent_maps[rank].items())       # guaranteed non-empty
            idx  = torch.tensor([rows, cols], device=device)
            vals = torch.ones(len(rows), device=device)
            mat  = torch.sparse_coo_tensor(
                idx, vals, (n_species, n_parents), device=device
            ).to_dense()
        else:
            # No variability in this rank → every species maps to parent 0
            mat = torch.zeros((n_species, n_parents), device=device)
            mat[:, 0] = 1.0
            print(f"[Info] Rank '{rank}' has no variability; using uniform mapping.")
        mapping_mats[rank] = mat

    # quick sanity-check
    for rank in RANK_ORDER:
        assert mapping_mats[rank].shape == (n_species, len(rank2lbl[rank]))

    # ------ load rank models ------------------------------------
    rank_models, rank_dims = {}, {}
    for rank, ckpt_dir in CKPT_DIRS.items():
        out_dim = len(rank2lbl[rank])
        net = build_model(enc_arch, "one_hot",
                          num_classifiers=num_heads,
                          encoder_path=None,
                          requires_grad=False,
                          output_dim=out_dim,
                          custom_trained=True).to(device)
        net.load_state_dict(torch.load(Path(ckpt_dir)/"model.ckpt",
                                       map_location=device))
        net.eval()
        rank_models[rank] = {"model": net, "out_dim": out_dim}
        if rank in RANK_ORDER:
            rank_dims[rank] = out_dim     # save dims for attention proj

    # ----------- feature helper ---------------------------------
    def build_feat_tensor(rank_logits_dict):
        return {r: torch.mean(torch.stack(rank_logits_dict[r], 2), 2)   # raw logits
                for r in RANK_ORDER}

    # ----------- train meta on train_df -------------------------
    model = MetaFusion(rank_dims, n_species).to(device)

    if META_WEIGHTS.exists():
        model.load_state_dict(torch.load(META_WEIGHTS, map_location=device))
        model.eval()
        print("Loaded meta-fusion weights.")
    else:
        loader_tr = build_dataloader(train_df, tfm, labelled=True)
        rank_logits, str_species, _ = collect_rank_logits(
            loader_tr, rank_models, device, num_heads, want_ids=False
        )

        feats = build_feat_tensor(rank_logits)    # dict rank -> (N,C_r)
        # stack in fixed order
        X = torch.cat([feats[r] for r in RANK_ORDER], dim=1)
        # NOTE: we keep dict form for forward; the concatenated X is only
        # for shuffling indices easier
        labels = torch.tensor(
            [rank2lbl["Species"][lab] for lab in str_species],
            device=device
        )

        opt = torch.optim.AdamW(model.parameters(), lr=LR_META, weight_decay=1e-2)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_META)

        for epoch in range(EPOCHS_META):
            perm = torch.randperm(X.size(0), device=device)
            running = 0.0
            for i in range(0, X.size(0), BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                batch_feats = {r: feats[r][idx] for r in RANK_ORDER}
                logits = model(batch_feats)

                # ---------- losses ----------
                ce_loss = F.cross_entropy(logits, labels[idx], label_smoothing=0.1)

                # ---------- hierarchy-consistency penalty ----------
                probs = logits.softmax(1)                       # (B, n_species)
                penalty = 0.0
                for rank in RANK_ORDER:
                    parent_prob    = torch.matmul(probs, mapping_mats[rank]) + 1e-9
                    tgt_parent  = parent_targets[rank][idx]     # (B,)
                    tgt_one_hot    = F.one_hot(tgt_parent, parent_prob.size(1)).float()
                    penalty       += F.kl_div(parent_prob.log(), tgt_one_hot, reduction="batchmean")

                # inside epoch loop, after penalty computed
                if epoch + 1 <= PENALTY_WARMUP_EPOCHS:
                    warm_coef = 0.0                                    # no penalty yet
                else:
                    # linear ramp over 5 epochs, then hold at λ
                    ramp = min(1.0, (epoch + 1 - PENALTY_WARMUP_EPOCHS) / 5)
                    warm_coef = ramp * LAMBDA_CONS
                loss = ce_loss + warm_coef * penalty


                opt.zero_grad(); loss.backward(); opt.step()
                running += loss.item() * idx.numel()
            sched.step()
            print(f"epoch {epoch+1}/{EPOCHS_META}"
                  f"  loss {running / X.size(0):.4f}")

        torch.save(model.state_dict(), META_WEIGHTS)
        model.eval()
        print("Saved meta-fusion weights.")

    # -------------------- inference -----------------------------
    test_df = pd.read_csv(TEST_CSV)
    loader_tst = build_dataloader(test_df, tfm, labelled=False)
    rk_logits, _, ann_ids = collect_rank_logits(
        loader_tst, rank_models, device, num_heads, want_ids=True
    )
    feats_tst = build_feat_tensor(rk_logits)

    with torch.no_grad():
        logits = model(feats_tst)
        pred_idx = logits.argmax(1).cpu()
        conf     = logits.softmax(1).max(1).values.cpu()

    preds = [idx2sp[i.item()] for i in pred_idx]
    save_predictions_to_csv(ann_ids, preds, conf.tolist(), OUT_CSV)
    print("Wrote predictions →", OUT_CSV.resolve())


if __name__ == "__main__":
    main()
