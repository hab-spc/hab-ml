"""Dataloader

#TODO DataLoader descriptio needed

"""
# Standard dist imports

# Third party imports
import logging
import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import OrderedDict

import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable

import skimage.util
import skimage.color
import scipy.ndimage
import scipy.misc

# Project level imports
from utils.config import opt
from utils.constants import *
from prepare_db.create_csv import create_proro_csv

# Module level constants
CLASSES = {0: 'Non-Prorocentrum', 1: 'Prorocentrum'}
NUM_CLASSES = len(CLASSES.keys())
DEVELOP = False

#TODO move all these functions to d_utils.py and import them over here
def pil_loader(path):
    return Image.open(path)

def numpy2tensor(x):
    if x.ndim == 3:
        x = np.transpose(x, (2, 0, 1))
    elif x.ndim == 4:
        x = np.transpose(x, (3, 0, 1, 2))
    return torch.from_numpy(x)

def tensor2numpy(x):
    return x.data.numpy()

def pil2numpy(x):
    return np.array(x).astype(np.float32)

def numpy2pil(x):
    mode = 'RGB' if x.ndim == 3 else 'L'
    return Image.fromarray(x, mode=mode)

def rgb_preproc(img):
    img = (2.*img[:, :, :3]/255. - 1).astype(np.float32)
    return img

def inverse_normalize():
    """#TODO Write function to un-normalize image to visualize
        During training, if we need to debug, we need to have this as an option
        This entails stuff like changing it back into numpy, BGR -> RGB,
        untransposing it. It won't necessarily include everything I just
        said, but to get an idea of what you gotta do to visualize the image again.
    """
    pass

def to_cuda(item, computing_device, label=False):
    """ Typecast item to cuda()

    Wrapper function for typecasting variables to cuda() to allow for
    flexibility between different types of variables (i.e. long, float)

    Loss function usually expects LongTensor type for labels, which is why
    label is defined as a bool.

    Computing device is usually defined in the Trainer()

    Args:
        item: Desired item. No specific type
        computing_device (str): Desired computing device.
        label (bool): Flag to convert item to long() or float()

    Returns:
        item
    """
    if label:
        item = Variable(item.to(computing_device)).long()
    else:
        item = Variable(item.to(computing_device)).float()
    return item

class SPCHABDataset(Dataset):
    """Custom Dataset class for the SPC Hab Dataset

    Current this is configured for the prorocentrum dataset...
    Expected dataset is stored in `phytoplankton-db`
    CSV files are located in a subdir `.../csv/`

    """
    def __init__(self, data_root, mode='train', input_size=112):
        """Initializes SPCHabDataset

        Args:
            data_root (str): Absolute path to the data csv files
            mode (str): Mode/partition of the dataset
            input_size (int): Image input size

        Attributes:
            data (Pandas.DataFrame): Dataframe of read CSV file
            classes (list): List of classes in str representation
            num_classes (int): Number of classes
            class_to_index(dict): Dictionary of each class with its
                associated images
        """
        assert mode in ('train', 'val'), 'mode should be train or val'
        self.mode = mode
        self.input_size = input_size
        self.rescale_size = input_size
        # PROROCENTRUM csv files
        csv_file = os.path.join(data_root,'proro_{}.csv').format(mode)
        self.data = pd.read_csv(csv_file)
        if DEVELOP:
            self.data = self.data.sample(n=100).reset_index(drop=True)

        # Clarify what transformations are needed here
        self.data_transform = {
            'train': transforms.Compose([transforms.Resize(self.rescale_size),
                                       transforms.RandomCrop(input_size),
                                       transforms.ColorJitter()]),
            'val':  transforms.Compose([transforms.Resize(self.rescale_size),
                                       transforms.CenterCrop(input_size)])
        }
        
        self.classes = sorted(self.data['label'].unique())
        self.num_class = len(self.classes)
        self.class_to_index = {}
        for cls in self.classes:
            # self.class_to_index[cls] = self.data.loc[self.data['label'] == cls, "images"]
            self.class_to_index[cls] = self.data.index[self.data['label'] == cls].tolist()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:

        """
        # Load image
        img_link = self.data.iloc[index]["images"]
        img = pil_loader(img_link)
        img = self.data_transform[self.mode](img)
        img = rgb_preproc(pil2numpy(img))
        img = numpy2tensor(img)

        target = self.data.iloc[index]["label"]
        target = self.encode_labels(target)

        id = self.data.iloc[index]['image_id']

        return {'rgb': img, 'label': target, 'id':id}
    
    def encode_labels(self, label):
        """ Encode labels given the enumerated class index

        Loss function from PyTorch expects labels to be in class indices,
        rather than one hot encodings.

        Args:
            label (str): Ground truth

        Returns:
            int: Class index

        """
        for idx, each in enumerate(self.classes):
            if each == label:
                cls_idx_lbl = idx
                return cls_idx_lbl

def get_dataloader(data_dir, batch_size=1, input_size=112, shuffle=True,
                   num_workers=4):
    """ Get the dataloader

    Args:
        data_dir (str): Absolute path to the csv data files
        batch_size (int): Batch size
        input_size (int): Image input size
        shuffle (bool): Flag for shuffling dataset
        num_workers (int): Number of workers

    Returns:
        dict: Dictionary holding each type of dataloader

    """
    # Create dataset if it hasn't been created
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print('Data dir not detected. Creating dataset @ {}'.format(data_dir))
        create_proro_csv(output_dir=data_dir, log2file=True)
    
    loader_dict = {}

    train_dataset = SPCHABDataset(data_dir, mode="train",
                                input_size=input_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers,
                                               pin_memory=True)
    loader_dict["train"] = train_loader

    val_dataset = SPCHABDataset(data_dir, mode="val",
                                input_size=input_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True)
    loader_dict["val"] = val_loader
    
    return loader_dict

if __name__ == '__main__':
    import time
    import sys
    """Example of running data loader"""
    batch_size = 16
    data_dir = '/data6/lekevin/hab-spc/phytoplankton-db/csv/proro1'
    loader = get_dataloader(data_dir=data_dir,
                            batch_size=batch_size, num_workers=2)["train"]
    print(len(loader.dataset))
    for i, batch in enumerate(loader):
        img = batch['rgb'].numpy()
        lbl = batch['label'].numpy()
        print(i, img.shape, img.min(), img.max(), img.dtype)
        print(i, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
        break