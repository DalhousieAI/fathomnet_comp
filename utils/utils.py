from utils.models import FathomNetModel, OneHotClassifier

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import (
    ExponentialLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

_BACKBONES = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "efficientnet-b0": models.efficientnet_b0,
    "efficientnet-b1": models.efficientnet_b1,
    "efficientnet-b2": models.efficientnet_b2,
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32,
}

_VIT_NUM_FEATURES = {
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
}

_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

def set_seed(seed, cudnn_deterministic=False):
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def map_label_to_idx(
        dataframe, 
        label_column,
        idx_column="label_idx"
        ):
    """
    Map labels to integers.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe with labels.
    label_column : str
        Column with labels.

    Returns
    -------
    pandas.DataFrame
        Dataframe with labels mapped to integers.
    dict
        Dictionary with labels mapped to integers.
    """
    labels = dataframe[label_column].unique()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    dataframe.loc[:, idx_column] = dataframe.loc[:, label_column].map(label_to_idx)
    return dataframe, label_to_idx

def build_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        ):
    """
    Build a PyTorch DataLoader.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def df_split(df, validation_ratio, seed):
    if validation_ratio <= 0 or validation_ratio >= 1:
        return df, None

    validation_size = int(len(df) * validation_ratio)

    # Split dataset into training and validation sets
    train_df, val_df = train_test_split(
        df, test_size=validation_size, random_state=seed, stratify=df["label_idx"]
    )
    return train_df, val_df

# Augmentation related functions
# (from FastAutoAug - MIT license)
# https://github.com/kakaobrain/fast-autoaugment
_IMAGENET_PCA = {
    "eigval": [0.2175, 0.0188, 0.0045],
    "eigvec": [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ],
}

# Lighting Class (from FastAutoAug - MIT license)
# https://github.com/kakaobrain/fast-autoaugment
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))

def convert_to_rgb(image):
    # Convert a 4-channel image (RGBA) to a 3-channel (RGB) image
    if image.size(0) == 4:
        return image[:3, :, :]  # Keep only the first 3 channels (R, G, B)
    return image

def get_augs(colour_jitter: bool, input_size=224, use_benthicnet=True):
    imagenet_mean_std = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    benthicnet_mean_std = transforms.Normalize(
        # labelled_dataset_stats
        # mean=[0.363, 0.420, 0.344], std=[0.207, 0.210, 0.183]
        mean=[0.359, 0.413, 0.386],
        std=[0.219, 0.215, 0.209],
    )

    if use_benthicnet:
        default_mean_std = benthicnet_mean_std
    else:
        default_mean_std = imagenet_mean_std

    if colour_jitter:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (input_size, input_size), interpolation=Image.BICUBIC
                ),
                # Crop settings may be too aggresive for biota
                transforms.RandomResizedCrop(
                    input_size, scale=(0.1, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                transforms.ToTensor(),
                transforms.Lambda(convert_to_rgb),
                Lighting(0.1, _IMAGENET_PCA["eigval"], _IMAGENET_PCA["eigvec"]),
                default_mean_std,
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(convert_to_rgb),
                default_mean_std,
            ]
        )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Lambda(convert_to_rgb),
            default_mean_std,
        ]
    )
    return train_transforms, val_transforms

# Model specific functions
def load_model_state(model, ckpt_path, origin="", component="encoder", verbose=0):
    # key = 'state_dict' for pre-trained models, 'model' for FB Imagenet
    alt_component_names = {
        "encoder": "backbone",
    }
    alt_component_name = alt_component_names.get(component, "")

    loaded_dict = torch.load(ckpt_path)

    if origin == "fb":
        key = "model"
    else:
        key = "state_dict"

    state = loaded_dict[key]
    loading_state = {}
    model_keys = model.state_dict().keys()

    if any(s in ckpt_path for s in ("mocov3", "mae", "vit")) and component == "encoder":
        if any(s in ckpt_path for s in ("hp", "hl", "hft")):
            loading_state = get_vit_state(
                model, state, model_keys, loading_state, reorder_pos_emb=False
            )
        else:
            loading_state = get_vit_state(model, state, model_keys, loading_state)
    else:
        for k in list(state.keys()):
            k_split = k.split(".")
            k_0 = k_split[0]
            if len(k_split) > 1:
                k_1 = k_split[1]
            else:
                k_1 = ""

            k_heads = ".".join([k_0, k_1])
            if k_0 == component or k_heads == component:
                k_to_check = k.replace(f"{component}.", "")
            elif k_0 == alt_component_name or k_heads == alt_component_name:
                k_to_check = k.replace(f"{alt_component_name}.", "")
            else:
                k_to_check = k

            if k_to_check in model_keys:
                loading_state[k_to_check] = state[k]
    if verbose > 0:
        print(
            f"Loading {len(loading_state.keys())} layers for {component}\n"
            " Expected layers (approx):\n\tViT-Base: 150\n\tViT-Large: 294\n\tResNet-50: 320"
        )
    model.load_state_dict(loading_state, strict=False)
    if verbose > 0:
        print(f"Loaded {component} from {ckpt_path}.")

    return model

# Function for supporting ViT loading
def get_vit_state(model, state, model_keys, loading_state, reorder_pos_emb=True):
    # Remove default ImageNet head from requiring loading
    model_keys = list(model_keys)[:-2]
    state_list = list(state.items())
    if reorder_pos_emb:
        pos_emb = state_list[1]
        conv_proj_w = state_list[2]
        conv_proj_b = state_list[3]

        state_list[1] = conv_proj_w
        state_list[2] = conv_proj_b
        state_list[3] = pos_emb

    for i, key in enumerate(model_keys):
        try:
            assert model.state_dict()[key].shape == state_list[i][1].shape
            loading_state[key] = state_list[i][1]
        except AssertionError:
            print(
                f"\nViT layer {i} {key}, does not match loading state layer {state_list[i][0]}"
            )
            print(
                f"Expected shape: {model.state_dict()[key].shape}, "
                f"from loading state got shape: {state_list[i][1].shape}"
            )
            continue
    return loading_state

# Freeze model weights
def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val

def build_model(
        encoder_arch, 
        classifier_type, 
        encoder_path=None, 
        requires_grad=True,
        output_dim=79
        ):
    enc = _BACKBONES[encoder_arch](weights="DEFAULT")
    if encoder_path:
        enc = load_model_state(enc, encoder_path)
    else:
        print("No encoder weights loaded.")
    set_requires_grad(enc, requires_grad)
    if "resnet" in encoder_arch:
        features_dim = enc.inplanes
        enc.fc = nn.Identity()
    elif "vit" in encoder_arch:
        features_dim = _VIT_NUM_FEATURES[encoder_arch]
        enc.heads = nn.Identity()

    if classifier_type == "one_hot":
        classifier = OneHotClassifier(features_dim, output_dim)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    model = FathomNetModel(enc, classifier)
    
    return model

def process_scheduler(
        optimizer,
        train_kwargs,
):
    if train_kwargs.scheduler_name == "warmup_cosine":
        scheduler = OneCycleLR(
            optimizer,
            pct_start=train_kwargs.warmup_epochs
            / train_kwargs.max_epochs,
            epochs=train_kwargs.max_epochs,
            steps_per_epoch=train_kwargs.steps_per_epoch,
            max_lr=train_kwargs.lr,
            div_factor=train_kwargs.lr / train_kwargs.warmup_start_lr
            if train_kwargs.warmup_epochs > 0
            else train_kwargs.lr,
            final_div_factor=train_kwargs.lr / train_kwargs.min_lr,
        )
    else:
        raise ValueError(
            f"Not implemented scheduler: {train_kwargs.scheduler_name}"
        )
    
    return scheduler

def train(
        model, 
        train_loader,
        val_loader,
        test_loader,
        label_map, 
        criterion, 
        device,
        train_kwargs,
        ):
    optimizer = _OPTIMIZERS[train_kwargs.optimizer_name](
        model.parameters(), 
        lr=train_kwargs.lr,
        weight_decay=train_kwargs.weight_decay,
    )

    scheduler = process_scheduler(optimizer, train_kwargs)

    model.train()

    training_losses = []
    training_accuracies = []
    if val_loader is not None:
        validation_losses = []
        validation_accuracies = []
    else:
        validation_losses = None
        validation_accuracies = None

    for epoch in range(train_kwargs.max_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = epoch_loss / len(train_loader.dataset)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{train_kwargs.max_epochs}], " + \
              f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        training_losses.append(epoch_loss)
        training_accuracies.append(accuracy)
        
        if val_loader is not None:
            val_loss, val_accuracy = test(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, " + \
                f"Validation Accuracy: {val_accuracy:.2f}%")
        
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)

        # Update the scheduler (after each epoch)
        scheduler.step()

    # Save the model checkpoint
    if train_kwargs.save_model:
        # Create folder for model if it doesn't exist,
        # based on model training settings
        model_path = f"./models/{train_kwargs.enc_arch}_pre-" + \
            f"{train_kwargs.enc_path}_cls-{train_kwargs.classifier_type}_" + \
            f"seed-{train_kwargs.seed}"

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(model.state_dict(), os.path.join(model_path, "model.ckpt"))

    # Save predictions on test set
    if test_loader is not None and label_map is not None:
        predictions = predict(model, test_loader, label_map, device)
        save_predictions_to_csv(predictions, os.path.join(model_path, "predictions.csv"))

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

def plot_training(
        training_losses, 
        training_accuracies, 
        validation_losses=None, 
        validation_accuracies=None,
        save_path=None,
    ):
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label="Training Loss", color="blue")
    if validation_losses is not None:
        plt.plot(validation_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(training_accuracies, label="Training Accuracy", color="blue")
    if validation_accuracies is not None:
        plt.plot(validation_accuracies, label="Validation Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    if save_path:
        plt.savefig(save_path)

def test(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = epoch_loss / len(val_loader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def predict(model, test_loader, label_map, device):
    invert_label_map = {v: k for k, v in label_map.items()}

    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # Convert predictions to labels using the label map
    predicted_labels = [
        invert_label_map[pred] for pred in predictions if pred in invert_label_map
    ]
    
    return predicted_labels

def save_predictions_to_csv(predictions, output_path):
    # Two columns: "annotation_id" and "concept_name"
    # Annotation IDs are the indices of the predictions
    df = pd.DataFrame({
        "annotation_id": range(len(predictions)),
        "concept_name": predictions,
    })
    df.to_csv(output_path, index=False)