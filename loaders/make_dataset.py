import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import pandas as pd


class DotsDataset:

    '''
    Dots Dataset
    '''

    def __init__(self, csv_file, gt_downsample=0, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.file = pd.read_csv(csv_file)
        self.root = './Dataset/'
        self.nSamples = len(self.file['counts'])

        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'

        csv_file = self.file

        cell_img_path = csv_file.iloc[idx, 1]
        dots_img_path = csv_file.iloc[idx, 2]

        cell_img = plt.imread(os.path.join(self.root, cell_img_path))
        dots_img = plt.imread(os.path.join(self.root, dots_img_path))

        # cell_img_tensor = torch.tensor(cell_img, dtype=torch.float)
        # dots_img_tensor = torch.tensor(dots_img, dtype=torch.float)

        # print(gt_dmap_tensor.sum())

        if self.transform is not None:
            cell_img_tensor = self.transform(cell_img)
            dots_img_tensor = self.transform(dots_img)

        return cell_img_tensor, dots_img_tensor


class DensityDataset:

    '''
    Density Dataset
    '''

    def __init__(self, csv_file, gt_downsample=0, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.file = pd.read_csv(csv_file)
        self.root = './Dataset/'
        self.nSamples = len(self.file['counts'])

        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'

        csv_file = self.file

        cell_img_path = csv_file.iloc[idx, 1]
        density_img_path = csv_file.iloc[idx, 3]

        # print(f'cell {cell_img_path}, root : {self.root}')
        # print(f'cell path : {os.path.join(self.root, cell_img_path)}')
        cell_img = plt.imread(os.path.join(self.root, cell_img_path))
        # print(f'density path : {os.path.join(self.root, density_img_path)}')
        density_img = Image.open(os.path.join(self.root, density_img_path))
        density_img = transforms.functional.to_grayscale(density_img)

        # cell_img_tensor = torch.tensor(cell_img, dtype=torch.float)
        # density_img_tensor = torch.tensor(density_img, dtype=torch.float)

        # print(gt_dmap_tensor.sum())

        if self.transform is not None:
            cell_img_tensor = self.transform(cell_img)
            density_img_tensor = self.transform(density_img)

        return cell_img_tensor, density_img_tensor


class CountDataset:

    '''
    Count Dataset
    '''

    def __init__(self, csv_file, gt_downsample=0, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        '''
        csv_file: csv file where paths to cell images an the associated number of cells are stored
        '''
        self.file = pd.read_csv(csv_file)
        self.root = './Dataset/'
        self.nSamples = len(self.file['counts'])

        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'

        csv_file = self.file

        cell_img_path = csv_file.iloc[idx, 1]
        counts = csv_file.iloc[idx, 4]

        cell_img = plt.imread(os.path.join(self.root, cell_img_path))

        # cell_img_tensor = torch.tensor(cell_img, dtype=torch.float)

        # print(gt_dmap_tensor.sum())

        if self.transform is not None:
            cell_img_tensor = self.transform(cell_img)

        return cell_img_tensor, counts
