# Code modified from: 
# https://github.com/DalhousieAI/benthicnet_probes/blob/master/utils/benthicnet_dataset.py
# Under GPL-3.0 License

import os

import PIL.Image
import torch.utils.data

class FathomNetDataset(torch.utils.data.Dataset):
    """FathomNet dataset."""

    def __init__(
        self,
        df,
        label_col="label_idx",
        transform=None,
        is_test=False,
    ):

        self.dataframe = df
        self.label_col = label_col
        self.transform = transform
        self.is_test = is_test


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        path = row["path"]

        sample = PIL.Image.open(path)

        if self.transform:
            sample = self.transform(sample)

        if self.is_test:
            annotation_id = path.split("_")[-1].split(".")[0]
            return sample, annotation_id
        
        return sample, row[self.label_col]
    
    def set_transform(self, transform):
        self.transform = transform

def custom_collate_fn(batch):
    """
    Custom collate function for FathomNetContextDataset.
    Each sample in batch is expected to be a tuple: (context, bbox, label)
    where bbox is a tensor of shape [4] in the format [x, y, w, h].
    This function converts each bbox to [batch_idx, x, y, x+w, y+h].
    """
    contexts, bboxes, labels = zip(*batch) # for testset, it is annotation_id
    # Stack the images (contexts) into one tensor.
    contexts = torch.stack(contexts, dim=0)
    
    # Process each bbox: add the corresponding batch index and convert [x, y, w, h]
    new_bboxes = []
    for idx, bbox in enumerate(bboxes):
        # bbox is [x, y, w, h]
        x, y, w, h = bbox
        # Convert to (batch_idx, x1, y1, x2, y2)
        new_bbox = torch.tensor([idx, x, y, x + w, y + h]).float()
        new_bboxes.append(new_bbox)
    new_bboxes = torch.stack(new_bboxes, dim=0)

    return contexts, new_bboxes, labels
    
class FathomNetContextDataset(torch.utils.data.Dataset):
    """FathomNet dataset."""

    def __init__(
        self,
        df,
        label_col="label_idx",
        transform=None,
        is_test=False,
    ):

        self.dataframe = df
        self.label_col = label_col
        self.transform = transform
        self.is_test = is_test


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        roi_path = row["path"]
        context_path = roi_path.replace("rois", "images")
        context_path = context_path.rsplit("_", 1)[0] + ".png"
        bbox = row["bbox"]

        context = PIL.Image.open(context_path)

        # Scale the bbox to match the resized context dimensions (224, 224).
        original_width, original_height = context.size
        scale_x = 224 / original_width
        scale_y = 224 / original_height

        scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]

        if self.transform:
            context = self.transform(context)

        if self.is_test:
            annotation_id = roi_path.split("_")[-1].split(".")[0]
            return context, scaled_bbox, annotation_id
        
        return context, scaled_bbox, row[self.label_col]
    
    def set_transform(self, transform):
        self.transform = transform

if __name__ == "__main__":
    from torchvision import transforms
    import pandas as pd
    df_dataset = pd.DataFrame({
            "path": ["/lustre07/scratch/ayyagari/cvpr_fgvc/data/train/rois/1_1.png",
                    "/lustre07/scratch/ayyagari/cvpr_fgvc/data/train/rois/2_2.png",
                    "/lustre07/scratch/ayyagari/cvpr_fgvc/data/train/rois/3_3.png",
                    "/lustre07/scratch/ayyagari/cvpr_fgvc/data/train/rois/4_4.png"],        
            "label_idx": ["Sebastolobus", "Holothuroidea", "Holothuroidea", "Keratoisis"],
            "bbox": [[0, 0, 100, 100], [10, 10, 110, 110], [20, 20, 120, 120], [30, 30, 130, 130]]
        })
    identity_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    dataset = FathomNetContextDataset(
            df=df_dataset,
            label_col="label_idx",
            transform=identity_transform,
            is_test=False
        )
