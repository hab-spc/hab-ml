"""CORAL Dataloader

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
from torch.utils.data import Dataset, ConcatDataset

# Project level imports
from data.d_utils import pil_loader, grab_classes
from data.label_encoder import HABLblEncoder
from utils.config import opt
from utils.constants import Constants as CONST
from utils.constants import SPCConstants as SPC_CONST
from utils.logger import Logger

# Module level constants
from data.d_utils import get_dataset_mean_and_std, compute_padding
from data.label_encoder import HABLblEncoder
from data.parse_data import DataParser
from data.transforms import DATA_TRANSFORMS
from utils.constants import Constants as CONST

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Coral_SPCHABDataset(Dataset, DataParser):
    def __init__(self, csv_file=None, mode=CONST.TRAIN, mean_std=None, transforms=None):
        # Initialize logging
        self.logger = logging.getLogger('dataloader_'+mode)
        self.logger.setLevel(opt.logging_level)

        # Initialize dataset attributes
        self.mode = mode
        self.transforms = transforms
        self.camera = None

        if self.mode != CONST.DEPLOY:
            self.camera = os.path.basename(os.path.dirname(csv_file))

            # get the datasets
        # === Read in the dataset ===#
        # options for reading datasets can be from a csv file or directory containing csv files.
        self.data = self.read_csv_dataset(csv_file)

        self.le = HABLblEncoder(mode=mode if self.camera != CONST.LAB else CONST.DEPLOY)
        self.classes, self.num_class = self.le.grab_classes(data=self.data)
        self.cls2idx, self.idx2cls = self.set_encode_decode(self.classes)
        self.le.fit(self.classes)

        self.default_unlabeled_idx = 0 #OTHER

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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


def get_coral_dataloader(csv_file=None, data_dir=None, camera=CONST.PIER,
                         mode=CONST.TRAIN, batch_size=1, shuffle=True,
                         num_workers=4):
    Logger.section_break(f'{mode.upper()} {camera.upper()} Dataset')
    logger = logging.getLogger('dataloader_' + mode)
    logger.setLevel(opt.logging_level)

    # Get the csv file
    if mode != CONST.DEPLOY:
        csv_file  = os.path.join(data_dir, '{}', '{}.csv')

    if mode == CONST.TRAIN:
        # Compute mean and std
        mean_std = get_dataset_mean_and_std(csv_file)

        # get the two datasets
        logger.info('Loading pier dataset...')
        pier_dataset = Coral_SPCHABDataset(csv_file=csv_file.format(CONST.PIER, mode),
                                      mean_std=mean_std,
                                      transforms=DATA_TRANSFORMS[mode])
        logger.debug(f'Pier Dataset Distribution\n{"-"*30}\n'
                     f'{pier_dataset.get_class_counts()}\n')
        logger.info('Loading lab dataset...')
        lab_dataset = Coral_SPCHABDataset(csv_file=csv_file.format(CONST.LAB, mode),
                                      mean_std=mean_std,
                                      transforms=DATA_TRANSFORMS[mode])
        logger.debug(f'Lab Dataset Distribution\n{"-"*30}\n'
                     f'{lab_dataset.get_class_counts()}\n')
        dataset = ConcatDataset(pier_dataset, lab_dataset)

    else:
        if mode == CONST.VAL:
            csv_file = csv_file.format(camera, mode)

        mean_std = {
            'pier': {
                'mean': [0.05021906002532712, 0.04498853271836354, 0.03423056766430313],
                'std':  [0.125469809183474, 0.10874887522563985, 0.07687470078134907]
            },
            'lab': {
                'mean': [0.06310804414032703, 0.05951638940097263, 0],
                'std':  [0.15778742164689152, 0, 0]
            }
        }

        dataset = Coral_SPCHABDataset(csv_file=csv_file, mean_std=mean_std,
                                      transforms=DATA_TRANSFORMS[mode])
        logger.debug(f'Dataset Distribution\n{"-"*30}\n{dataset.get_class_counts()}\n')

    logger.info('Dataset size: {}'.format(dataset.__len__()))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=_init_fn)
    return data_loader

def _init_fn(worker_id):
    np.random.seed(int(opt.SEED))

if __name__ == '__main__':
    from data.d_utils import to_cuda
    opt.data_dir = 'DB/coral_c34_workshop2019'
    opt.mean_std_json = os.path.join(opt.data_dir, 'mean_std.json')

    opt.model_dir = '../experiments/alexnet_coral_c34_workshop2019/'

    train_data_loader = get_coral_dataloader(data_dir=opt.data_dir, camera='pier',
                                        batch_size=opt.batch_size,
                                        mode=CONST.TRAIN)

    src_data_loader = get_coral_dataloader(data_dir=opt.data_dir, camera='pier',
                                        batch_size=opt.batch_size,
                                        mode=CONST.VAL)
    target_data_loader = get_coral_dataloader(data_dir=opt.data_dir, camera='lab',
                                        batch_size=opt.batch_size,
                                        mode=CONST.VAL)

    computing_device = torch.device("cuda")

    for i, batch in enumerate(train_data_loader):

        src_batch, target_batch = batch[0], batch[1]

        # process batch items: images, labels
        src_img = to_cuda(src_batch[CONST.IMG], computing_device)
        src_label = to_cuda(src_batch[CONST.LBL], computing_device, label=True)

        target_img = to_cuda(target_batch[CONST.IMG], computing_device)

        print(src_label)
