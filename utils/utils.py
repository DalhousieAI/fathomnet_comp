# MCLoss and get_constr_out function
# from Constrained Feed-Forward Neural Network for HML
# (Coherent Hierarchical Multi-Label Classification Networks - GPL-3.0 License)
# https://github.com/EGiunchiglia/C-HMCNN

from utils.models import FathomNetModel, \
    OneHotClassifier, ConstrainedFFNNModel, \
    MultiHeadClassifier

import json
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
    "efficientnet-v2-s": models.efficientnet_v2_s,
    "efficientnet-v2-m": models.efficientnet_v2_m,
    "efficientnet-v2-l": models.efficientnet_v2_l,
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32,
    "vit_h_14": models.vit_h_14,
    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2,
}

_VIT_NUM_FEATURES = {
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
    "vit_h_14": 1280,
}

_EFFICIENTNET_NUM_FEATURES = {
    "efficientnet-b0": 1280,
    "efficientnet-b1": 1280,
    "efficientnet-b2": 1408,
    "efficientnet_v2_l": 1280,
}

_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

_COST_MATS_PATHS = {
    "dist": "./cfg/hierarchy/score_table.npy",
    "cce": "./cfg/hierarchy/score_table.npy",
    "ncce": "./cfg/hierarchy/norm_score_table.npy",
    "cce1": "./cfg/hierarchy/score_table_ones.npy",
    "ncce1": "./cfg/hierarchy/norm_score_table_ones.npy",
}

def get_cost_matrix(mode):
    path = _COST_MATS_PATHS[mode]
    cost_matrix = np.load(path)

    return torch.from_numpy(cost_matrix).float()


def read_json(file_path):
    # Expects a python object in the json file
    # e.g. {"key": "value"}
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
    
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

def get_augs(colour_jitter: bool, input_size=224, use_benthicnet=False):
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
                transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
                # Use random resized crop for variation
                transforms.RandomResizedCrop(
                    input_size, scale=(0.1, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                # Randomly apply color jitter (standard brightness/contrast/saturation augmentation)
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)],
                    p=0.8,
                ),
                # Randomly choose between a perspective distortion and an affine transformation (zoom out effect)
                transforms.RandomChoice(
                    [
                        transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
                        transforms.RandomAffine(degrees=0, scale=(0.8, 1.0)),
                    ]
                ),
                # Randomly apply an extra crop with padding
                transforms.RandomApply(
                    [transforms.RandomCrop(input_size, padding=4)],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Lambda(convert_to_rgb),
                # Apply lighting noise (AlexNet-style PCA noise)
                Lighting(0.1, _IMAGENET_PCA["eigval"], _IMAGENET_PCA["eigvec"]),
                default_mean_std,
                # Randomly apply random erasing
                transforms.RandomApply(
                    [transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3))],
                    p=0.5,
                ),
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
def load_model_state(
        model, 
        ckpt_path, 
        origin="", 
        component="encoder", 
        custom_trained=True, 
        verbose=0
        ):
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

    if custom_trained:
        state = loaded_dict
    else:
        state = loaded_dict[key]

    loading_state = {}
    model_keys = model.state_dict().keys()

    if any(s in ckpt_path for s in ("mocov3", "mae", "vit")) and component == "encoder":
        if any(s in ckpt_path for s in ("hp", "hl", "hft")) or custom_trained:
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
        num_classifiers=1, 
        encoder_path=None, 
        requires_grad=True,
        output_dim=79,
        custom_trained=False,
        ):
    enc = _BACKBONES[encoder_arch](weights="DEFAULT")
    if encoder_path:
        enc = load_model_state(enc, encoder_path, custom_trained=custom_trained)
    else:
        print("No encoder weights loaded.")
    set_requires_grad(enc, requires_grad)
    if "resnet" in encoder_arch:
        features_dim = enc.inplanes
        enc.fc = nn.Identity()
    elif "vit" in encoder_arch:
        features_dim = _VIT_NUM_FEATURES[encoder_arch]
        enc.heads = nn.Identity()
    elif "efficientnet" in encoder_arch:
        features_dim = _EFFICIENTNET_NUM_FEATURES[encoder_arch]
        enc.classifier = nn.Identity()

    classifier = MultiHeadClassifier(
        classifier_type=classifier_type,
        num_classifiers=num_classifiers,
        features_dim=features_dim,
        output_dim=output_dim,
    )

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

def get_constr_out(x, R):
    # Given the output of the neural network 
    # x returns the output of MCM given the hierarchy constraint 
    # expressed in the matrix R
    
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out

def mcloss(logits, targets, R, criterion):
    # MCLoss - why doubles?
    constr_output = get_constr_out(logits, R)
    output = targets * logits.double()
    output = get_constr_out(output, R)
    output = (1 - targets) * constr_output.double() + targets * output
    # Print the indices where targets are 1
    loss = criterion(output, targets)

    return loss

def predict_batch(outputs, k=2, eps=1e-8, requires_grad=False):
    if requires_grad:
        in_outputs = outputs
    else:
        in_outputs = torch.stack([output.detach() for output in outputs])

    softmax = nn.Softmax(dim=-1)
    softmax_outputs = softmax(in_outputs)

    avg_output = torch.mean(softmax_outputs, dim=0)
    top_k_means = torch.topk(avg_output, k=k, dim=-1, sorted=True)[0]

    pred_mean_prob, predicted = torch.max(avg_output.data, 1)

    if softmax_outputs.shape[0] == 1:
        constrained_pcs = torch.ones_like(predicted, dtype=torch.float32)
    else:
        std_output = torch.std(softmax_outputs, dim=0)
        top_k_stds = torch.topk(std_output, k=k, dim=1)[0]

        pcs_factor_num = (top_k_means[:, 0] - top_k_means[:, 1])
        pcs_factor_denom = top_k_stds[:, 0] + top_k_stds[:, 1]
        pcs_exponent = pcs_factor_num / (pcs_factor_denom + eps)
        
        constrained_pcs = 1 - torch.exp(-pcs_exponent)
    
    return predicted, constrained_pcs*pred_mean_prob

def train(
        model, 
        train_loader,
        val_loader,
        test_loader,
        label_map, 
        criterion,
        dist_metric, 
        device,
        train_kwargs,
        ):
    
    optimizer = _OPTIMIZERS[train_kwargs.optimizer_name](
        model.parameters(), 
        lr=train_kwargs.lr,
        weight_decay=train_kwargs.weight_decay,
    )

    one_hot_cond = "one_hot" in train_kwargs.classifier_type
    is_conf = "conf" in train_kwargs.classifier_type    

    scheduler = process_scheduler(optimizer, train_kwargs)

    model.train()

    training_losses = []
    training_confidences = []
    training_dists = []
    if one_hot_cond:
        training_accuracies = []
    else:
        training_accuracies = None
        assert train_kwargs.hierarchy_dict is not None, \
            "Hierarchy dict is required for HML classifier."

    if val_loader is not None:
        validation_losses = []
        if one_hot_cond:
            validation_accuracies = []
            validation_confidences = []
            validation_dists = []
        else:
            validation_accuracies = None
            validation_confidences = None
            validation_dists = None
    else:
        validation_losses = None
        validation_accuracies = None
        validation_confidences = None
        validation_dists = None
        
    if one_hot_cond:
        R = None
    else:
        R = train_kwargs.descendent_matrix.to(device)

    for epoch in range(train_kwargs.max_epochs):
        epoch_loss = 0.0
        epoch_confidences = []
        epoch_dists = []
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if one_hot_cond:
                predicted, confidence = predict_batch(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                batch_dist = dist_metric(
                    predictions=predicted,
                    targets=labels,
                )

                epoch_dists.append(batch_dist)

                batch_confidence = torch.mean(confidence).item()
                epoch_confidences.append(batch_confidence)
                if is_conf:
                    losses = torch.stack(
                        [criterion(output, labels, confidence) for output in outputs]
                        )
                else:
                    losses = torch.stack([criterion(output, labels) for output in outputs])
            else:
                losses = []
                labels = labels.double()
                for output in outputs:
                    loss = mcloss(
                        logits=output,
                        targets=labels,
                        R=R,
                        criterion=criterion,
                    )
                    losses.append(loss)
                losses = torch.stack(losses)

            losses = torch.mean(losses)
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item() * images.size(0)

        epoch_loss = epoch_loss / len(train_loader.dataset)
        
        if one_hot_cond:
            accuracy = 100 * correct / total
            mean_epoch_confidence = np.mean(epoch_confidences)
            mean_epoch_dist = np.mean(epoch_dists)
            
            print(f"Epoch [{epoch+1}/{train_kwargs.max_epochs}], " + \
                f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, " + \
                f"Avg. Confidence: {mean_epoch_confidence:.2f} " + \
                f"Avg. Distance: {mean_epoch_dist:.2f}")
            
            training_accuracies.append(accuracy)
            training_confidences.append(mean_epoch_confidence)
            training_dists.append(mean_epoch_dist)
        else:
            print(f"Epoch [{epoch+1}/{train_kwargs.max_epochs}], " + \
                f"Loss: {epoch_loss:.4f}")
        training_losses.append(epoch_loss)
        
        if val_loader is not None:
            val_loss, val_accuracy, val_confidence, val_dist = test(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                dist_metric=dist_metric,
                device=device,
                one_hot_cond=one_hot_cond,
                is_conf=is_conf,
                R=R,
            )
            if one_hot_cond:
                print(f"Validation Loss: {val_loss:.4f}, " + \
                    f"Validation Accuracy: {val_accuracy:.2f}% " + \
                    f"Avg. Confidence: {val_confidence:.2f} " + \
                    f"Avg. Distance: {val_dist:.2f}")
                validation_accuracies.append(val_accuracy)
                validation_confidences.append(val_confidence)
                validation_dists.append(val_dist)
            else:
                print(f"Validation Loss: {val_loss:.4f}")
            validation_losses.append(val_loss)

        # Update the scheduler (after each epoch)
        scheduler.step()

    # Save the model checkpoint
    if train_kwargs.save_model:
        # Create folder for model if it doesn't exist,
        # based on model training settings
        if train_kwargs.enc_path is not None:
            enc_path = train_kwargs.enc_path.split("/")[-1].split(".")[0]
        else:
            enc_path = "None"
        model_path = f"./models/{train_kwargs.enc_arch}_pre-" + \
            f"{enc_path}_cls-{train_kwargs.classifier_type}_" + \
            f"rank-{train_kwargs.rank}_" + \
            f"seed-{train_kwargs.seed}_e-{train_kwargs.max_epochs}_aug-{train_kwargs.use_colour_jitter}_isz-{train_kwargs.input_size}" + \
            f"lr-{train_kwargs.lr}_n-heads-{train_kwargs.num_classifiers}"

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(model.state_dict(), os.path.join(model_path, "model.ckpt"))

    # Save predictions on test set
    if test_loader is not None and label_map is not None:
        if one_hot_cond:
            annotation_ids, predictions, confidences = predict(
                model, 
                test_loader, 
                label_map, 
                device
                )
        else:
            annotation_ids, predictions = predict_hml(
                model, 
                test_loader, 
                label_map,
                train_kwargs.hierarchy_dict,
                R,
                device,
                mode="simple",
            )
            confidence = None
        save_predictions_to_csv(
            annotation_ids, 
            predictions,
            confidences, 
            os.path.join(model_path, "predictions.csv")
            )

        print(f"Predictions saved to {os.path.join(model_path, 'predictions.csv')}")

    # Save training curves
    if train_kwargs.save_curves:
        plot_training(
            training_losses, 
            training_accuracies,
            training_confidences,
            training_dists, 
            validation_losses, 
            validation_accuracies,
            validation_confidences,
            validation_dists,
            save_path=os.path.join(model_path, "training_curves.png")
        )

        print(f"Training curves saved to {os.path.join(model_path, 'training_curves.png')}")

def plot_training(
        training_losses, 
        training_accuracies,
        training_confidences=None,
        training_dists=None, 
        validation_losses=None, 
        validation_accuracies=None,
        validation_confidences=None,
        validation_dists=None,
        save_path=None,
    ):
    plt.figure(figsize=(12, 12))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(training_losses, label="Training Loss", color="blue")
    if validation_losses is not None:
        plt.plot(validation_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot training accuracy
    if training_accuracies is not None:
        plt.subplot(2, 2, 2)
        plt.plot(training_accuracies, label="Training Accuracy", color="blue")
        if validation_accuracies is not None:
            plt.plot(validation_accuracies, label="Validation Accuracy", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Training and Validation Accuracy")
        plt.legend()

    if training_confidences is not None:
        plt.subplot(2, 2, 3)
        plt.plot(training_confidences, label="Training Confidence", color="blue")
        if validation_confidences is not None:
            plt.plot(validation_confidences, label="Validation Confidence", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Confidence")
        plt.title("Training and Validation Confidence")
        plt.legend()

    if training_dists is not None:
        plt.subplot(2, 2, 4)
        plt.plot(training_dists, label="Training Distance", color="blue")
        if validation_dists is not None:
            plt.plot(validation_dists, label="Validation Distance", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Distance")
        plt.title("Training and Validation Distance")
        plt.legend()

    if save_path:
        plt.savefig(save_path)

def test(
        model, 
        val_loader, 
        criterion,
        dist_metric, 
        device, 
        one_hot_cond=False,
        is_conf=False,
        R=None
        ):
    model.eval()
    epoch_loss = 0.0
    epoch_confidences = []
    epoch_dists = []
    if one_hot_cond:
        correct = 0
        total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if one_hot_cond:
                predicted, confidence = predict_batch(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                batch_dist = dist_metric(
                    predictions=predicted,
                    targets=labels,
                )

                epoch_dists.append(batch_dist)

                batch_confidence = torch.mean(confidence).item()
                epoch_confidences.append(batch_confidence)

                if is_conf:
                    losses = torch.stack(
                        [criterion(output, labels, confidence) for output in outputs]
                        )
                else:
                    losses = torch.stack([criterion(output, labels) for output in outputs])
            else:
                losses = []
                labels = labels.double()
                for output in outputs:
                    loss = mcloss(
                        logits=output,
                        targets=labels,
                        R=R,
                        criterion=criterion,
                    )
                    losses.append(loss)
                losses = torch.stack(losses)

            losses = torch.mean(losses)
            epoch_loss += losses.item() * images.size(0)
                
    epoch_loss = epoch_loss / len(val_loader.dataset)

    if one_hot_cond:
        accuracy = 100 * correct / total
        mean_epoch_confidence = np.mean(epoch_confidences)
        mean_epoch_dist = np.mean(epoch_dists)
        return epoch_loss, accuracy, mean_epoch_confidence, mean_epoch_dist
    else:
        return epoch_loss, None, None, None

def predict(model, test_loader, label_map, device):
    invert_label_map = {v: k for k, v in label_map.items()}

    model.eval()
    annotation_ids = []
    predictions = []
    image_names = []
    confidences = []
    with torch.no_grad():
        for images, annotation_id in test_loader:
            images = images.to(device)
            outputs = model(images)

            predicted, confidence = predict_batch(outputs)
            annotation_ids.extend(annotation_id)
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
    
    # Convert predictions to labels using the label map
    predicted_labels = [
        invert_label_map[pred] for pred in predictions
    ]
    
    return annotation_ids, predicted_labels, confidences

def predict_hml(
        model, 
        test_loader, 
        label_map,
        hierarchical_dict,
        R,
        device,
        mode="simple",
        ):
    # Get the predictions from the model
    model.eval()
    annotation_ids = []
    predictions = []

    # Get the leaf nodes from the hierarchical dict
    leaf_nodes = []
    for key, value in hierarchical_dict.items():
        if len(value) == 1:
            leaf_nodes.append(int(key))
    # sort the leaf nodes
    leaf_nodes.sort()

    with torch.no_grad():
        for images, annotation_id in test_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = get_constr_out(outputs, R)
            batch_len = len(images)
            if mode == "simple":
                # check predictions that are leaf nodes
                predicted = outputs[:, leaf_nodes]
                # Get max index for each row
                predicted = torch.argmax(predicted, dim=1)
                # Get the original leaf node index
                leaf_nodes_tensor = torch.tensor(leaf_nodes).to(device)
                leaf_nodes_tensor = leaf_nodes_tensor.unsqueeze(0).repeat(batch_len, 1)
                predicted = leaf_nodes_tensor[torch.arange(batch_len), predicted]
            else:
                raise ValueError(
                    f"Unknown mode: {mode}. Use 'simple'."
                )
            annotation_ids.extend(annotation_id)
            predictions.extend(predicted.cpu().numpy())
    # Get the corresponding label from the label map
    # You are here!
    predicted_labels = [label_map[str(int(pred))] for pred in predictions]

    return annotation_ids, predicted_labels
    

def save_predictions_to_csv(annotation_ids, predictions, confidences, output_path):
    # Two columns: "annotation_id" and "concept_name"
    # Annotation IDs are the indices of the predictions
    df = pd.DataFrame({
        "annotation_id": annotation_ids,
        "concept_name": predictions,
        "confidence": confidences,
    })
    df.to_csv(output_path, index=False)

# Hierarchical functions
def convert_indices_to_label(indices, array_len=192):
    new_array = np.zeros(array_len, dtype=int)
    for i in indices:
        new_array[int(i)] = 1
    return new_array

# For each row, collect the values of the heirarchical headers into a list if they are not null
def collect_hierarchy(
        row, 
        heirarchical_headers=[
            "phylum", 
            "class", 
            "order", 
            "family", 
            "genus", 
            "species"
            ]
        ):
    return [row[header] for header in heirarchical_headers if pd.notnull(row[header])]

def get_hierarchy_from_df(df):
    # Get the hierarchy from the datafrom with annotations
    # domain,kingdom,phylum,class,order,family,genus,species
    df["hierarchy"] = df.apply(
        lambda row: collect_hierarchy(row), axis=1
    )
    parent_child_dict = {}
    for _, row in df.iterrows():
        hierarchical_annotation = row["hierarchy"]
        for i in range(len(hierarchical_annotation)):
            parent = hierarchical_annotation[i]
            children = hierarchical_annotation[i:]
            if parent not in parent_child_dict:
                parent_child_dict[parent] = []
            else:
                parent_child_dict[parent].extend(children)
    # Turn everything in the dict to ints including keys
    # Convert keys to int
    parent_child_dict = {int(k): v for k, v in parent_child_dict.items()}
    # Convert values to int
    for parent in parent_child_dict:
        parent_child_dict[parent] = [int(x) for x in parent_child_dict[parent]]
    # Remove duplicates in the list of children
    for parent in parent_child_dict:
        parent_child_dict[parent] = list(set(parent_child_dict[parent]))
    descendent_matrix = convert_hierarchy_dict_to_descendent_matrix(
        parent_child_dict
    )
    return parent_child_dict, descendent_matrix

def convert_hierarchy_dict_to_descendent_matrix(
        parent_child_dict, 
        ):
    num_classes = len(parent_child_dict)
    descendent_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for parent, children in parent_child_dict.items():
        for child in children:
            descendent_matrix[parent][child] = 1
    return descendent_matrix