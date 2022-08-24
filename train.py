from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import numpy as np
from model import Acrnn
import pickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os
import torch
import pdb
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from dataset import IEMOCAPDataset
import argparse

num_classes = 4
traindata_path = './IEMOCAP.pkl'
validdata_path = 'inputs/valid.pkl'
checkpoint = './checkpoint'
model_name = 'best_model.pth'
clip = 0


def train(args):
    #####load data##########

    train_dataset = IEMOCAPDataset(df_csv_train, feature_file, transform=ToTensor())
    val_dataset = IEMOCAPDataset(df_csv_val, feature_file, transform=ToTensor())
    test_dataset = IEMOCAPDataset(df_csv_test, feature_file, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    ##########tarin model###########

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)
        elif type(m) == torch.nn.Conv2d:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    model = Acrnn()
    model.apply(init_weights)

    num_epoch = 250

    trainer = pl.Trainer(devices=1,auto_select_gpus=True, limit_train_batches=128, max_epochs=num_epoch)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_df', type=str)
    args = parser.parse_args()
    train(args)
