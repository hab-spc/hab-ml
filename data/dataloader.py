"""Dataloader

#TODO DataLoader description needed

"""
import logging
# Standard dist imports
import os

# Third party imports
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

# Project level imports
from data.d_utils import pil_loader, compute_padding
from data.label_encoder import HABLblEncoder
from data.transforms import DATA_TRANSFORMS
from data.parse_data import DataParser
from utils.config import opt
from utils.constants import Constants as CONST
from utils.constants import SPCConstants as SPC_CONST
from utils.logger import Logger

# Module level constants
#TODO make this dictionary dynamic according to the data loaded

CLASSES = {}
UNKNOWN = {999: 'Unknown'}
NUM_CLASSES = len(CLASSES.keys())

class SPCHABDataset(Dataset, DataParser):
    """Custom Dataset class for the SPC Hab Dataset

    Current this is configured for the prorocentrum dataset...
    Expected dataset is stored in `phytoplankton-db`
    CSV files are located in a subdir `.../csv/`

    """
    def __init__(self, csv_file=None, mode='train', transforms=None):
        """Initializes SPCHabDataset

        Args:
            csv_file (str): Absolute path to the csv dir. If in
                deploy mode it should be the absolute path to the csv file
                itself
            mode (str): Mode/partition of the dataset
            input_size (int): Image input size

        Attributes:
            data (Pandas.DataFrame): Dataframe of read CSV file
            classes (list): List of classes in str representation
            num_classes (int): Number of classes
            class_to_index(dict): Dictionary of each class with its
                associated images
        """
        assert mode in (CONST.TRAIN, CONST.VAL, CONST.DEPLOY), 'mode: train, val, deploy'
        self.mode = mode
        self.transforms = transforms

        Logger.section_break(f'{self.mode.upper()} Dataset')
        self.logger = logging.getLogger('dataloader_'+mode)
        self.logger.setLevel(opt.logging_level)
        
        # === Read in the dataset ===#
        # options for reading datasets can be from a csv file or directory containing csv files.

        # If deployment, create dataset from csv file
        # Else (training or validation) access csv file given data directory and mode
        self.data = pd.read_csv(csv_file)

        self.le = HABLblEncoder(mode=mode)
        self.classes, self.num_class = self.le.grab_classes(data=self.data)
        self.cls2idx, self.idx2cls = self.set_encode_decode(self.classes)
        self.le.fit(self.classes)

        self.default_unlabeled_idx = 0 # OTHER


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:

        """
        # get image file path, label, and image_identifier
        img = self.data.iloc[index][CONST.IMG]
        target = self.data.iloc[index][CONST.LBL]
        id = self.data.iloc[index][CONST.IMG]


        # Load image
        img = pil_loader(img)
        # Apply data augmentation
        if self.transforms != None:
            # Padding to Square
            img = transforms.Pad(compute_padding(img.size),
                                 fill=0, padding_mode='constant')(img)
            img = self.transforms(img)

        # Encode label
        if not isinstance(target, (int, np.int64)):
            target = self.encode_labels(target)

        return {CONST.IMG: img, CONST.LBL: target, CONST.ID:id}
    
    def encode_labels(self, label):
        if label in self.cls2idx:
            return self.cls2idx[label]
        else:
            return self.default_unlabeled_idx

    def get_class_counts(self):
        return self.data[CONST.LBL].value_counts()


def get_dataloader(data_dir=None, csv_file=None, batch_size=1, shuffle=True,
                   num_workers=4, mode=CONST.TRAIN):
    """ Get the dataloader

    Function accepts either data directory or csv file to create a dataloader

    Args:
        mode (str):
        data_dir (str): Relative path to the csv data files
        csv_file (str): Absolute path of the csv file
        batch_size (int): Batch size
        input_size (int): Image input size
        shuffle (bool): Flag for shuffling dataset
        num_workers (int): Number of workers

    Returns:
        dict: Dictionary holding each type of dataloader

    """
    logger = logging.getLogger('dataloader_' + mode)
    logger.setLevel(opt.logging_level)

    if mode != CONST.DEPLOY:
        csv_file = os.path.join(opt.data_dir, '{}.csv'.format(mode))

    dataset = SPCHABDataset(csv_file=csv_file, mode=mode,
                                    transforms=DATA_TRANSFORMS[mode])

    logger.debug(f'Dataset Distribution\n{"-" * 30}\n'
                 f'{dataset.get_class_counts()}\n')
    logger.info('Dataset size: {}'.format(dataset.__len__()))

    # Create the dataloader from the given dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers,
                                               pin_memory=True)
    return data_loader

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

if __name__ == '__main__':
    DEBUG_DATALODER = False
    DEBUG_SPCHAB = False
    SPICI = False

    """Example of running data loader"""
    if DEBUG_SPCHAB:
        if SPICI:
            deploy_data = '/data6/lekevin/hab-master/hab-spc/data/experiments/test_deploy_classifier.csv'
        else:
            deploy_data = '/data6/phytoplankton-db/hab_invitro/images/20190515_static_html/images/00000'
        data_loader = get_dataloader(mode=CONST.DEPLOY, data_dir=deploy_data,
                                     batch_size=opt.batch_size,
                                     input_size=opt.input_size)
        print(len(data_loader))
        for i, batch in enumerate(data_loader):
            img = batch['images'].numpy()
            lbl = batch['label'].numpy()
            print(i, img.shape, img.min(), img.max(), img.dtype)
            print(i, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
            print(batch['image_id'])


