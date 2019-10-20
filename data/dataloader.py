"""Dataloader

#TODO DataLoader descriptio needed

"""
# Standard dist imports
from collections import OrderedDict
import os
import sys
import logging

# Third party imports
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

# Project level imports
from data.prepare_db import create_proro_csv, prepare_db, create_lab_csv
from utils.config import opt
from utils.constants import *

# Module level constants
#TODO make this dictionary dynamic according to the data loaded

CLASSES = {}
UNKNOWN = {999: 'Unknown'}
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
    

def create_class_dict (lbs_all_classes):
    """
    input: all classes labels which are unsorted
    Helper Function: Sort class labels and assign them to global CLASSES vairable
    """
    global CLASSES
    lbs_all_classes.sort()
    CLASSES = {}
    c = 0
    for i in lbs_all_classes:
        CLASSES[c] = i
        c+=1
    NUM_CLASSES = len(CLASSES.keys())
    opt.class_num = len(lbs_all_classes)
    
    
def grab_classes (mode, df_unique = None, filename = None):
    """
    Fill in CLASSES glob variable depends on different modes
    if in train mode, create CLASSES out of train.csv.
    if in val/deploy mode, retrieve CLASSES from the given filename
    """
    if mode == TRAIN:
        lbs_all_classes = df_unique
    else:
        lbs_all_classes = []
        
        #check if file is able to be opened
        try:
            f = open(filename,'r')
        except Error as e:
            logger.debug(e)
            sys.exit()
            
        with open(filename,'r') as f:
                label_counts = f.readlines()
        label_counts = label_counts[:-1]
        for i in label_counts:
            class_counts = i.strip()
            class_counts = class_counts.split()
            class_name = ''
            for j in class_counts:
                if not j.isdigit():
                    class_name += (' '+j)
            class_name = class_name.strip()
            lbs_all_classes.append(class_name)
    create_class_dict (lbs_all_classes)
        
        
class SPCHABDataset(Dataset):
    """Custom Dataset class for the SPC Hab Dataset

    Current this is configured for the prorocentrum dataset...
    Expected dataset is stored in `phytoplankton-db`
    CSV files are located in a subdir `.../csv/`

    """
    def __init__(self, data_root, mode='train', input_size=224):
        """Initializes SPCHabDataset

        Args:
            data_root (str): Absolute path to the csv dir. If in
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
        assert mode in (TRAIN, VAL, DEPLOY), 'mode: train, val, deploy'
        self.mode = mode
        self.input_size = input_size
        self.rescale_size = input_size
        self.logger = logging.getLogger('dataloader_'+mode)
        
        # PROROCENTRUM csv files
        #TODO refactor code to accept abs path to csv_file rather than
        # assembling with data_root
        if self.mode == DEPLOY:
            if os.path.isdir(data_root):
                csv_file = create_lab_csv(data_root)
            else:
                csv_file = data_root # Absolute path to the deploy_data
            deploy_prep = True
        else:
            csv_file = os.path.join(data_root,'{}.csv').format(mode)
        self.data = pd.read_csv(csv_file)

        
        #get classes
        filename = os.path.join(opt.model_dir, 'train_data.info')
        if self.mode != DEPLOY:
            df_unique = self.data[SPCData.LBL].unique()
        else:
            df_unique = None
        grab_classes (self.mode, df_unique = df_unique, filename = filename)
        self.logger.info('All classes detected are: '+str(CLASSES))
        self.logger.info('opt.class_num = '+str(opt.class_num))
        
        #Store data_info
        if opt.mode != DEPLOY:
            data_save_path = os.path.join(opt.model_dir, mode+'_data.info')
            with open(data_save_path, 'w') as f:
                f.write(str(self.data[SPCData.LBL].value_counts()))

        if DEVELOP:
            self.data = self.data.sample(n=100).reset_index(drop=True)

        if self.mode == DEPLOY and opt.lab_config == True:
            if self.check_SPCformat():
                image_dir = self._get_image_dir(data_root)
                self.data = prepare_db(data=self.data, image_dir=image_dir,
                                       csv_file=csv_file)

        # Clarify what transformations are needed here
        self.data_transform = {
            TRAIN: transforms.Compose([transforms.Resize(self.rescale_size),
                                       transforms.CenterCrop(input_size),
                                      transforms.RandomAffine(360, translate=(0.1, 0.1), scale=None, shear=None, resample=False, fillcolor=0),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                      ]),
            VAL:  transforms.Compose([transforms.Resize(self.rescale_size),
                                       transforms.CenterCrop(input_size),
                                         transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                     ]),
            DEPLOY: transforms.Compose([transforms.Resize(self.rescale_size),
                                       transforms.CenterCrop(input_size),
                                       transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                        ])
        }
        
        self.classes = sorted(CLASSES.values())
        self.num_class = len(self.classes)
        if mode in [TRAIN, VAL]:
            self.class_to_index = {}
            for cls in self.classes:
                self.class_to_index[cls] = self.data.index[self.data[SPCData.LBL] ==
                                                           cls].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:

        """
        # Load image
        img = pil_loader(self.data.iloc[index][SPCData.IMG])
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

        if SPCData.LBL not in self.data.columns:
            target = self.data.iloc[index]['user_labels']
        else:
            target = self.data.iloc[index][SPCData.LBL]
        if not isinstance(target, (int, np.int64)):
            target = self.encode_labels(target)

        if opt.lab_config == True:
            if SPCData.ID in self.data.columns.values:
                id = self.data.iloc[index][SPCData.ID]
            else:
                id = 0
        else:
            if SPCData.IMG in self.data.columns.values:
                id = self.data.iloc[index][SPCData.IMG]
            else:
                id = 0


        return {SPCData.IMG: img, SPCData.LBL: target, SPCData.ID:id}
    
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
        self.logger.debug('No such class named ' + label + ' exist! ')


    def check_SPCformat(self, prepare_db_flag=False):
        """Check if SPCFormat for dataset preparation

        #check if dir or file
        if directory
        """
        class_constants = [value for name, value in vars(SPCData).items()
                           if not name.startswith('__')]
        for col_name in class_constants:
            if col_name not in self.data.columns.values:
                prepare_db_flag = True
                break

        if prepare_db_flag:
            return True
        else:
            return False

    def _get_image_dir(self, data_root):
        """Get image dir"""
        #TODO fix this function for grabbing the image directory
        master_db_dir = '/data6/lekevin/hab-master/hab-spc/phytoplankton-db'
        master_db_dir = '/data6/phytoplankton-db/hab_in_situ/hab_field'
        img_dir = os.path.basename(data_root).split('.')[0]
        img_dir = os.path.join(master_db_dir, img_dir)
        if os.path.isdir(img_dir):
            return img_dir
        else:
            raise OSError(f'{img_dir} does not exist')


def get_dataloader(data_dir, batch_size=1, input_size=224, shuffle=True,
                   num_workers=4, mode=TRAIN):
    """ Get the dataloader

    Args:
        mode (str):
        data_dir (str): Absolute path to the csv data files
        batch_size (int): Batch size
        input_size (int): Image input size
        shuffle (bool): Flag for shuffling dataset
        num_workers (int): Number of workers

    Returns:
        dict: Dictionary holding each type of dataloader

    """
    # Create dataset if it hasn't been created
    if mode in [TRAIN, VAL]:
        if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
            logger.debug('Data dir not detected. Creating dataset @ {}'.format(data_dir))
            create_proro_csv(output_dir=data_dir, log2file=True)
    else:
        if not os.path.exists(data_dir):
            raise ValueError('File does not exist')

    dataset = SPCHABDataset(data_dir, mode=mode,
                                input_size=input_size)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers,
                                               pin_memory=True)
    return data_loader

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
        data_loader = get_dataloader(mode=DEPLOY, data_dir=deploy_data,
                                     batch_size=opt.batch_size,
                                     input_size=opt.input_size)
        print(len(data_loader))
        for i, batch in enumerate(data_loader):
            img = batch['images'].numpy()
            lbl = batch['label'].numpy()
            print(i, img.shape, img.min(), img.max(), img.dtype)
            print(i, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
            print(batch['image_id'])


