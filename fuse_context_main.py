from types import SimpleNamespace

import argparse
import os
import numpy as np
import pandas as pd
import torch

from utils.dataset import FathomNetContextDataset, custom_collate_fn
from utils.models import AttentionFusion, FusedClassifier
from utils.context_utils import train_epoch, load_bioclip, predict
from utils.utils import df_split, save_predictions_to_csv, plot_training,\
    map_label_to_idx, set_seed, \
    read_json

def add_bbox(df, bbox_file):
    data = read_json(bbox_file)
    annotations = data.get("annotations", [])
    images = {}
    for annotation in annotations:
        images[str(annotation["image_id"]) + "_" + str(annotation["id"])+".png"] = annotation["bbox"]
    
    # Add bboxes as a new column in df using matching keys
    df["bbox"] = df.apply(
        lambda row: images.get(row["path"].split("/")[-1], None),
        axis=1
    )
    return df

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


    # df = pd.read_csv("../data/train/annotations.csv")
    df = pd.read_csv("cfg/hierarchy/hierarchy_labels_train_noNone.csv")
    df, label_map = map_label_to_idx(df, train_kwargs.rank)
    # add bbox coords   
    df = add_bbox(df, "cfg/context/dataset_train.json")

    test_df = pd.read_csv("../data/test/annotations.csv")
    test_df = add_bbox(test_df, "cfg/context/dataset_test.json")
    
    label_col = "label_idx"

    train_df, val_df = df_split(
        df, validation_ratio=train_kwargs.validation_ratio, seed=train_kwargs.seed
    )

    context_model, preprocess = load_bioclip(device=device)
    train_dataset = FathomNetContextDataset(
        df=train_df, 
        label_col=label_col,
        transform=preprocess,
        )

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_kwargs.batch_size,
            shuffle=True,
            num_workers=train_kwargs.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )


    print("Total samples:", len(df))
    print(len(train_dataset), f"training samples, {len(train_dataset)/len(df):.2%} of total")

    train_kwargs.steps_per_epoch = len(train_dataloader)

    output_dim = len(label_map)
    criterion = torch.nn.CrossEntropyLoss()

    C = 768 # BioCLIP ViT‑B/16 output dim
    fusion = AttentionFusion(dim=C, num_heads=4).to(device)
    classifier = FusedClassifier(C, output_dim).to(device)

    backbone_params = [p for p in context_model.parameters() if p.requires_grad]
    head_params     = list(fusion.parameters()) + list(classifier.parameters())

    optimizer = torch.optim.AdamW(
        [{'params': head_params,     'lr': 3e-4},
        {'params': backbone_params, 'lr': 3e-5}],
        betas=(0.9,0.999), weight_decay=1e-2
    )    

    training_losses, training_accuracies, validation_losses, validation_accuracies = [], [], [], []
    for epoch in range(1, train_kwargs.max_epochs + 1):
        loss, acc = train_epoch(train_dataloader, context_model, fusion,
                                classifier, criterion, optimizer, device)
        print(f"epoch {epoch:02d}  loss {loss:.4f}  acc {acc:.3f}")
        training_losses.append(loss)
        training_accuracies.append(acc)
        #TODO: add valid loss and acc
        
    
    # Save the model checkpoint
    if train_kwargs.save_model:
        # Create folder for model if it doesn't exist,
        # based on model training settings
        model_path = f"./models/fused_bioclip_roialign_" + \
            f"rank-{train_kwargs.rank}_"+ \
            f"seed-{train_kwargs.seed}_e-{train_kwargs.max_epochs}_isz-{train_kwargs.input_size}"

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(classifier.state_dict(), os.path.join(model_path, "model.ckpt"))
    
    
    test_dataset = FathomNetContextDataset(
        df=test_df, 
        label_col=label_col,
        transform=preprocess,
        is_test=True
        )

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_kwargs.batch_size,
            shuffle=False,
            num_workers=train_kwargs.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    if test_loader is not None and label_map is not None:
        annotation_ids, predictions = predict(test_loader,
                                                context_model,
                                                fusion,
                                                classifier,
                                                label_map,
                                                device
                                                )
    
    save_predictions_to_csv(
            annotation_ids, 
            predictions, 
            os.path.join(model_path, "predictions.csv")
            )
    print(f"Predictions saved to {os.path.join(model_path, 'predictions.csv')}")
    
    # Save training curves
    if train_kwargs.save_curves:
        plot_training(
            training_losses, 
            training_accuracies, 
            validation_losses, 
            validation_accuracies,
            save_path=os.path.join(model_path, "training_curves.png")
        )

        print(f"Training curves saved to {os.path.join(model_path, 'training_curves.png')}")

if __name__ == "__main__":
    main()