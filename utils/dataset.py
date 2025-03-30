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
            return sample
        
        return sample, row[self.label_col]
    
    def set_transform(self, transform):
        self.transform = transform
