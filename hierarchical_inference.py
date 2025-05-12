import torch
import torch.nn.functional as F

import pandas as pd
from typing import Dict, List


import os
from pathlib import Path

from utils.utils import build_model, convert_to_rgb, map_label_to_idx, save_predictions_to_csv
from utils.dataset import FathomNetDataset
import torchvision.transforms as transforms

OUT_CSV     = Path("species_predictions.csv")
def predict(model, test_loader, label_map, device):
    invert_label_map = {v: k for k, v in label_map.items()}

    model.eval()
    annotation_ids = []
    predictions = []
    with torch.no_grad():
        for images, annotation_id in test_loader:
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()  # Compute softmax values for all classes
            # Map softmax outputs to string labels using invert_label_map
            mapped_predictions = [
                {invert_label_map[idx]: value for idx, value in enumerate(softmax_output)}
                for softmax_output in softmax_outputs
            ]
            annotation_ids.append(annotation_id)
            predictions.append(mapped_predictions)  # Store mapped predictions
    
    return annotation_ids, predictions
    

if __name__ == "__main__":

    rank_ckpt = {"Phylum": "models/vit_l_16_pre-None_cls-one_hot_rank-Phylum_seed-42_e-100_aug-True_isz-224",
                 "Class": "models/vit_l_16_pre-model_cls-one_hot_rank-Class_seed-42_e-20_aug-True_isz-224",
                 "Order": "models/vit_l_16_pre-model_cls-one_hot_rank-Order_seed-42_e-20_aug-True_isz-224",
                 "Family": "models/vit_l_16_pre-model_cls-one_hot_rank-Family_seed-42_e-20_aug-True_isz-224",
                 "Genus": "models/vit_l_16_pre-model_cls-one_hot_rank-Genus_seed-42_e-20_aug-True_isz-224",
                 "Species": "models/vit_l_16_pre-model_cls-one_hot_rank-Species_seed-42_e-20_aug-True_isz-224"}
    # Load the model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    imagenet_mean_std = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(convert_to_rgb),
            imagenet_mean_std,
        ])
    
    test_df = pd.read_csv("../data/test/annotations.csv")
    train_df = pd.read_csv("cfg/hierarchy/hierarchy_labels_train_noNone.csv")

    test_dataset = FathomNetDataset(
            df=test_df, 
            transform=val_transforms,
            is_test=True,
            )

    test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

    # 1.  build label maps for *every* rank
    rank2label: Dict[str, Dict[str, int]] = {}
    for rank in rank_ckpt.keys():
        _, rank2label[rank] = map_label_to_idx(train_df, rank)
    n_species = len(rank2label["Species"])

    # 2.  pre‑compute ancestor‑index maps  (species_idx -> ancestor_idx at each higher rank)
    #      e.g.   anc_idx["Phylum"][species_idx]  = 5
    anc_idx = {rank: torch.zeros(n_species, dtype=torch.long) for rank in rank_ckpt if rank != "Species"}
    for sp_name, sp_idx in rank2label["Species"].items():
        row = train_df[train_df["Species"] == sp_name].iloc[0]
        for rank in anc_idx:
            anc_name                       = row[rank]
            anc_idx[rank][sp_idx]          = rank2label[rank][anc_name]
    
    # 3.  storage for per‑rank probabilities
    N = len(test_df)
    rank_probs: Dict[str, torch.Tensor] = {}

    
    for rank, model_dir in rank_ckpt.items():
        output_dim = len(rank2label[rank])
        ckpt_path = os.path.join(model_dir, "model.ckpt")
        model = build_model(
            encoder_arch="vit_l_16",
            encoder_path=None,
            classifier_type="one_hot",
            output_dim = output_dim,
            requires_grad=False,
        ).to(device)
        
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    
        all_probs = torch.zeros(N, output_dim, device=device)
        pos = 0
        annotation_ids = []
        with torch.no_grad():
            for imgs, ann_ids in test_dataloader:             # _ids ignored
                bs  = imgs.size(0)
                logits = model(imgs.to(device))
                all_probs[pos:pos+bs] = F.softmax(logits, dim=1)
                pos += bs
                annotation_ids.extend(ann_ids)

        rank_probs[rank] = all_probs
        del model
        torch.cuda.empty_cache()

    # 5.  build *leaf‑posterior*  p(species | x)
    log_leaf_post = rank_probs["Species"].log()       # (N, S)
    for rank, idx_vec in anc_idx.items():             # skip Species itself
        log_leaf_post += rank_probs[rank][:, idx_vec].log()

    leaf_post = log_leaf_post.exp()
    leaf_post /= leaf_post.sum(dim=1, keepdim=True)   # normalise

    # 6.  Bayes‑risk (0/1 loss)  → arg‑max posterior
    pred_idx  = leaf_post.argmax(dim=1)               # (N,)
    idx2species = {v: k for k, v in rank2label["Species"].items()}
    preds = [idx2species[i.item()] for i in pred_idx]

    # 7.  write CSV
    save_predictions_to_csv(annotation_ids, preds, OUT_CSV)
