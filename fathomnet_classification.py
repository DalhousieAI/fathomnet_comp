import pandas as pd
import torch
from types import SimpleNamespace

from utils.dataset import FathomNetDataset
from utils.utils import (
    build_model, 
    df_split, 
    get_augs, 
    map_label_to_idx, 
    set_seed, 
    train
)

# Global train_kwargs declaration
train_kwargs = None

def load_data():
    df = pd.read_csv("../data/train/annotations.csv")
    test_df = pd.read_csv("../data/test/annotations.csv")
    return df, test_df

def prepare_dataframes(df):
    global train_kwargs
    df, label_map = map_label_to_idx(df, "label")
    train_df, val_df = df_split(df, validation_ratio=train_kwargs.validation_ratio, seed=train_kwargs.seed)
    return train_df, val_df, label_map

def prepare_augmentations():
    global train_kwargs
    train_augs, val_augs = get_augs(
        colour_jitter=False, 
        use_benthicnet=train_kwargs.use_benthicnet_normalization
    )
    return train_augs, val_augs

def prepare_datasets(train_df, val_df, test_df, train_augs, val_augs):
    global train_kwargs
    train_dataset = FathomNetDataset(
        df=train_df, 
        label_col="label_idx",
        transform=train_augs,
    )
    # Only create a validation dataset if a validation split exists
    if train_kwargs.validation_ratio > 0:
        val_dataset = FathomNetDataset(
            df=val_df, 
            label_col="label_idx",
            transform=val_augs,
        )
    else:
        val_dataset = []
    test_dataset = FathomNetDataset(
        df=test_df, 
        label_col="label_idx",
        transform=val_augs,
        is_test=True,
    )
    return train_dataset, val_dataset, test_dataset

def prepare_dataloaders(train_dataset, val_dataset, test_dataset):
    global train_kwargs
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_kwargs.batch_size,
        shuffle=True,
        num_workers=train_kwargs.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if train_kwargs.validation_ratio > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_kwargs.batch_size,
            shuffle=False,
            num_workers=train_kwargs.num_workers,
            pin_memory=True
        )
    else:
        val_loader = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_kwargs.batch_size,
        shuffle=False,
        num_workers=train_kwargs.num_workers,
        pin_memory=True
    )
    # Update train_kwargs with steps_per_epoch.
    train_kwargs.steps_per_epoch = len(train_loader)
    return train_loader, val_loader, test_loader

def build_and_train_model(train_loader, val_loader, test_loader, label_map, device):
    global train_kwargs
    if train_kwargs.classifier_type == "one_hot":
        output_dim = len(label_map)
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

    criterion = torch.nn.CrossEntropyLoss()

    train(
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_map=label_map,
        criterion=criterion, 
        device=device,
        train_kwargs=train_kwargs,
        
    )

def main():
    global train_kwargs
    # Define training parameters
    train_kwargs_dict = {
        "enc_arch": "vit_b_32",
        "enc_path": None,
        "classifier_type": "one_hot",
        "seed": 0,
        "cudnn_deterministic": False,
        "batch_size": 64,
        "num_workers": 4,
        "validation_ratio": 0.,
        "fine_tune": True,
        "optimizer_name": "adamw",
        "lr": 1e-4,
        "warmup_start_lr": 1e-6,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "scheduler_name": "warmup_cosine",
        "max_epochs": 500,
        "warmup_epochs": 10,
        "save_model": True,
        "save_curves": True,
        "use_benthicnet_normalization": False,
    }
    train_kwargs = SimpleNamespace(**train_kwargs_dict)

    # Set seed for reproducibility
    set_seed(train_kwargs.seed, cudnn_deterministic=train_kwargs.cudnn_deterministic)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, test_df = load_data()
    train_df, val_df, label_map = prepare_dataframes(df)
    train_augs, val_augs = prepare_augmentations()
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_df, val_df, test_df, train_augs, val_augs)
    train_loader, val_loader, test_loader = prepare_dataloaders(train_dataset, val_dataset, test_dataset)

    print("Total samples:", len(df))
    print(len(train_dataset), f"training samples, {len(train_dataset)/len(df):.2%} of total")
    if train_kwargs.validation_ratio > 0:
        print(len(val_dataset), f"validation samples, {len(val_dataset)/len(df):.2%} of total")

    build_and_train_model(train_loader, val_loader, test_loader, label_map, device)

if __name__ == "__main__":
    main()