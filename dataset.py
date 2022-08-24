import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from h5py import File
import pandas as pd
import numpy as np


class IEMOCAPDataset(Dataset):
    def __init__(self, df_csv, feature_file, transform=None):
        self.df_csv = pd.read_csv(df_csv, sep=',')
        self._dataset = None
        self.feature_file = feature_file
        self.transform = transform

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, idx):
        if self._dataset is None:
            self._dataset = File(self.feature_file, 'r')

        audio_id = self.df_csv.iloc[idx]['wav_file']
        spec = np.expand_dims(self._dataset[f"{audio_id}/spec"][()], axis=0)
        delta1 = np.expand_dims(self._dataset[f"{audio_id}/delta1"][()], axis=0)
        delta2 = np.expand_dims(self._dataset[f"{audio_id}/delta2"][()], axis=0)
        label = self._dataset[f"{audio_id}/label"][()]
        data = np.concatenate((spec, delta1, delta2), axis=0)
        if self.transform:
            data = self.transform(data)
        return data, label
