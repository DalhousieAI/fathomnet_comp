from types import SimpleNamespace

import os
import argparse
import pandas as pd
import torch

from utils.dataset import FathomNetDataset
from utils.temperature_scaling import ModelWithTemperature
from utils.utils import build_model, df_split, get_augs,\
    map_label_to_idx, \
    read_json, \
    predict, save_predictions_to_csv
parser = argparse.ArgumentParser(description="Read a JSON file and load it as a dictionary.")
# Add arguments
parser.add_argument(
    '--cal_cfg', 
    type=str, 
    required=True, 
    help='Path to the JSON file for training configuration.'
)
args = parser.parse_args()

cal_kwargs = read_json(args.cal_cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cal_kwargs = SimpleNamespace(**cal_kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = cal_kwargs.model_path
ckpt_path = os.path.join(model_path, "model.ckpt")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Model path {ckpt_path} does not exist.")
model = build_model(
    encoder_arch=cal_kwargs.enc_arch,
    classifier_type=cal_kwargs.classifier_type,
    requires_grad=cal_kwargs.fine_tune,
)

# Load the checkpoint state dict
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)

df = pd.read_csv("../data/train/annotations.csv")
test_df = pd.read_csv("../data/test/annotations.csv")

df, label_map = map_label_to_idx(df, "label")

label_col = "label_idx"

train_df, val_df = df_split(
    df, validation_ratio=cal_kwargs.validation_ratio, seed=cal_kwargs.seed
)
_, val_augs = get_augs(
    colour_jitter=cal_kwargs.use_colour_jitter,
    input_size=cal_kwargs.input_size, 
    use_benthicnet=cal_kwargs.use_benthicnet_normalization
    )

val_dataset = FathomNetDataset(
        df=val_df, 
        label_col=label_col,
        transform=val_augs,
        )

val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cal_kwargs.batch_size,
        shuffle=False,
        num_workers=cal_kwargs.num_workers,
        pin_memory=True
    )

print("Train Dataset Class Distribution:")
print(train_df["label_idx"].value_counts())

print("Validation Dataset Class Distribution:")
print(val_df["label_idx"].value_counts())

scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(val_dataloader, save_debug_plots=True)

# Save predictions on test set
test_dataset = FathomNetDataset(
        df=test_df, 
        label_col=label_col,
        transform=val_augs,
        is_test=True,
        )

test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cal_kwargs.batch_size,
        shuffle=False,
        num_workers=cal_kwargs.num_workers,
        pin_memory=True
    )

if test_dataloader is not None and label_map is not None:
    annotation_ids, predictions, confidences = predict(scaled_model, test_dataloader, label_map, device)
    save_predictions_to_csv(
        annotation_ids, 
        predictions, 
        os.path.join(model_path, "predictions_withconf.csv"),
        confidences=confidences
        )

    print(f"Predictions saved to {os.path.join(model_path, 'predictions_withconf.csv')}")