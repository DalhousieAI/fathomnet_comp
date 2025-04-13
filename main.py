from types import SimpleNamespace

import argparse
import json
import numpy as np
import pandas as pd
import torch

from utils.dataset import FathomNetDataset
from utils.utils import build_model, df_split, get_augs, \
    map_label_to_idx, set_seed, collect_hierarchy, \
    read_json, train, convert_indices_to_label

def main():
    parser = argparse.ArgumentParser(description="Read a JSON file and load it as a dictionary.")
    # Add arguments
    parser.add_argument(
        '--train_cfg', 
        type=str, 
        required=True, 
        help='Path to the JSON file for training configuration.'
    )
    args = parser.parse_args()

    train_kwargs = read_json(args.train_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_kwargs = SimpleNamespace(**train_kwargs)

    # Set seed for reproducibility
    set_seed(train_kwargs.seed, cudnn_deterministic=train_kwargs.cudnn_deterministic)

    is_hml = train_kwargs.classifier_type == "hml"

    df = pd.read_csv("./data/train/annotations.csv")
    test_df = pd.read_csv("./data/test/annotations.csv")

    df, label_map = map_label_to_idx(df, "label")

    label_col = "label_idx"
    if is_hml:
        label_map = json.load(
            open("./data/train/index_to_taxon.json", "r")
        )
        assert train_kwargs.hierarchy_dict_path is not None, (
            "hierarchy_dict_path must be specified for HML classifier."
        )
        train_kwargs.hierarchy_dict = json.load(
            open(train_kwargs.hierarchy_dict_path, "r")
        )
        
        label_col = "label_hml"
        df[label_col] = df.apply(collect_hierarchy, axis=1)
        df[label_col] = df[label_col].apply(convert_indices_to_label)

    train_df, val_df = df_split(
        df, validation_ratio=train_kwargs.validation_ratio, seed=train_kwargs.seed
    )

    train_augs, val_augs = get_augs(
        colour_jitter=False, 
        use_benthicnet=train_kwargs.use_benthicnet_normalization
        )

    train_dataset = FathomNetDataset(
        df=train_df, 
        label_col=label_col,
        transform=train_augs,
        )

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_kwargs.batch_size,
            shuffle=True,
            num_workers=train_kwargs.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    if train_kwargs.validation_ratio > 0:
        val_dataset = FathomNetDataset(
            df=val_df, 
            label_col=label_col,
            transform=val_augs,
            )

        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=train_kwargs.batch_size,
                shuffle=False,
                num_workers=train_kwargs.num_workers,
                pin_memory=True
            )
    else:
        val_dataset = []
        val_dataloader = None

    test_dataset = FathomNetDataset(
        df=test_df, 
        label_col=label_col,
        transform=val_augs,
        is_test=True,
        )

    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_kwargs.batch_size,
            shuffle=False,
            num_workers=train_kwargs.num_workers,
            pin_memory=True
        )

    print("Total samples:", len(df))
    print(len(train_dataset), f"training samples, {len(train_dataset)/len(df):.2%} of total")
    print(len(val_dataset), f"validation samples, {len(val_dataset)/len(df):.2%} of total")

    train_kwargs.steps_per_epoch = len(train_dataloader)

    if train_kwargs.classifier_type == "one_hot":
        output_dim = len(label_map)
        criterion = torch.nn.CrossEntropyLoss()
    elif train_kwargs.classifier_type == "hml":
        assert train_kwargs.descendent_matrix_path is not None, (
            "descendent_matrix_path must be specified for HML classifier."
        )
        descendent_matrix = torch.from_numpy(
            np.load(train_kwargs.descendent_matrix_path)
        ).to(device)
        output_dim = descendent_matrix.shape[0]
        train_kwargs.descendent_matrix = descendent_matrix
        criterion = torch.nn.BCELoss()
    else:
        raise ValueError("Unsupported classifier type.")

    model = build_model(
        encoder_arch=train_kwargs.enc_arch,
        encoder_path=train_kwargs.enc_path,
        classifier_type=train_kwargs.classifier_type,
        requires_grad=train_kwargs.fine_tune,
        output_dim=output_dim,
    )

    model = model.to(device)

    train(
        model=model, 
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        label_map=label_map,
        criterion=criterion, 
        device=device,
        train_kwargs=train_kwargs,
        )

if __name__ == "__main__":
    main()