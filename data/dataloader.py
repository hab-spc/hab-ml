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
from data.d_utils import pil_loader, grab_classes
from data.label_encoder import HABLblEncoder
from utils.config import opt
from utils.constants import Constants as CONST
from utils.constants import SPCConstants as SPC_CONST
from utils.logger import Logger

# Module level constants
#TODO make this dictionary dynamic according to the data loaded

CLASSES = {}
UNKNOWN = {999: 'Unknown'}
NUM_CLASSES = len(CLASSES.keys())

class SPCHABDataset(Dataset):
    """Custom Dataset class for the SPC Hab Dataset

    Current this is configured for the prorocentrum dataset...
    Expected dataset is stored in `phytoplankton-db`
    CSV files are located in a subdir `.../csv/`

    """
    def __init__(self, csv_file=None, data_dir=None, mode='train', input_size=224):
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
        self.input_size = input_size
        self.rescale_size = input_size
        Logger.section_break(f'{self.mode.upper()} Dataset')
        self.logger = logging.getLogger('dataloader_'+mode)
        self.logger.setLevel(opt.logging_level)
        
        # === Read in the dataset ===#
        # options for reading datasets can be from a csv file or directory containing csv files.
        if self.mode != CONST.DEPLOY:
            csv_file = os.path.join(data_dir, '{}.csv').format(mode)

        # If deployment, create dataset from csv file
        # Else (training or validation) access csv file given data directory and mode
        self.data = pd.read_csv(csv_file)

        # get classes and save it
        classes_fname = os.path.join(opt.model_dir, '{}_data.info'.format(self.mode))
        if self.mode == CONST.TRAIN or self.mode == CONST.VAL:
            df_unique = self.data[SPC_CONST.LBL].unique()

            with open(classes_fname, 'w') as f:
                f.write(str(self.data[SPC_CONST.LBL].value_counts()))

            # gets classes based off dataframe
            self.classes = opt.classes = grab_classes(self.mode, df_unique=df_unique)
        else:
            # gets classes based off the train_data.info that is written during training
            self.classes = opt.classes = grab_classes(self.mode, filename=classes_fname)

        self.num_class = opt.class_num = len(self.classes)
        self.logger.info('All classes detected are: '+str(self.classes))
        self.logger.info('opt.class_num = '+str(opt.class_num))

        # Clarify what transformations are needed here
        self.data_transform = {
            CONST.TRAIN: transforms.Compose([transforms.Resize(self.rescale_size),
                                             transforms.CenterCrop(input_size),
                                             transforms.RandomAffine(360, translate=(0.1, 0.1), scale=None, shear=None,
                                                                     resample=Image.BICUBIC, fillcolor=0),
                                             transforms.ToTensor(),
                                             # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                             ]),
            CONST.VAL: transforms.Compose([transforms.Resize(self.rescale_size),
                                           transforms.CenterCrop(input_size),
                                           transforms.ToTensor(),
                                           # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                           ]),
            CONST.DEPLOY: transforms.Compose([transforms.Resize(self.rescale_size),
                                              transforms.CenterCrop(input_size),
                                              transforms.ToTensor(),
                                              # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                              ])
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:

        """
        # Load image
        img = pil_loader(self.data.iloc[index][SPC_CONST.IMG])
        ### Padding to Square
        img_size = img.size
        img_padding = [0,0]
        if img_size[0] > img_size[1]:
            img_padding[1] = int((img_size[0]-img_size[1])/2)
        else:
            img_padding[0] = int((img_size[1]-img_size[0])/2)
        img_padding = tuple(img_padding)
        img = transforms.Pad(img_padding,fill=0, padding_mode='constant')(img)
        ###
        img = self.data_transform[self.mode](img)

        if SPC_CONST.LBL not in self.data.columns:
            target = self.data.iloc[index]['user_labels']
        else:
            target = self.data.iloc[index][SPC_CONST.LBL]

        if not isinstance(target, (int, np.int64)):
            target = self.encode_labels(target)

        if opt.lab_config == True:
            if SPC_CONST.ID in self.data.columns.values:
                id = self.data.iloc[index][SPC_CONST.ID]
            else:
                id = 0
        else:
            if SPC_CONST.IMG in self.data.columns.values:
                id = self.data.iloc[index][SPC_CONST.IMG]
            else:
                id = 0

        return {SPC_CONST.IMG: img, SPC_CONST.LBL: target, SPC_CONST.ID: id}
    
    def encode_labels(self, label):
        """ Encode labels given the enumerated class index

        Loss function from PyTorch expects labels to be in class indices,
        rather than one hot encodings.

        Args:
            label (str): Ground truth

        Returns:
            int: Class index

        """
        cls_idx_lbl = 0
        for idx, each in enumerate(self.classes):
            if each == label:
                cls_idx_lbl = idx
        return cls_idx_lbl

def get_dataloader(data_dir=None, csv_file=None, batch_size=1, input_size=224, shuffle=True,
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

    # Create a dataset from the training and validation csv files (consolidated in one data directory)
    logger = logging.getLogger('dataloader')
    if mode in [CONST.TRAIN, CONST.VAL] and data_dir:
        if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
            logger.error('Data dir not detected. Need to create dataset @ {}'.format(data_dir))
            raise NotADirectoryError('Directory does not exist')

        else:
            dataset = SPCHABDataset(data_dir=data_dir, mode=mode,
                                    input_size=input_size)

    # Else create dataset from the given csv file (assumed for evaluation/deployment use case)
    else:
        if not os.path.exists(csv_file):
            raise ValueError('File does not exist')

        else:
            dataset = SPCHABDataset(csv_file=csv_file, mode=mode,
                                    input_size=input_size)

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


