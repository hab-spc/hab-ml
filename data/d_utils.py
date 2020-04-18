""" """
import logging
import sys
# Standard dist imports

# Third party imports
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from PIL import Image

# Project level imports
from utils.config import opt
from utils.constants import Constants as CONST


# Module level constants

def read_csv_dataset(csv_file, verbose=False):
    df = pd.read_csv(csv_file)
    if verbose:
        print('\n{0:*^80}'.format(' Reading in the dataset '))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.info())
        print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
        print(df.head())
    return df

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
    img = (2. * img[:, :, :3] / 255. - 1).astype(np.float32)
    return img

def compute_padding(img_size):
    img_padding = [0, 0]
    if img_size[0] > img_size[1]:
        img_padding[1] = int((img_size[0] - img_size[1]) / 2)
    else:
        img_padding[0] = int((img_size[1] - img_size[0]) / 2)
    img_padding = tuple(img_padding)
    return img_padding

def inverse_normalize():
    """#TODO Write function to un-normalize image to visualize
        During training, if we need to debug, we need to have this as an option
        This entails stuff like changing it back into numpy, BGR -> RGB,
        untransposing it. It won't necessarily include everything I just
        said, but to get an idea of what you gotta do to visualize the image again.
    """
    pass


def grab_classes(mode, df_unique=None, filename=None):
    """
    Fill in CLASSES glob variable depends on different modes
    if in train mode, create CLASSES out of train.csv.
    if in val/deploy mode, retrieve CLASSES from the given filename
    """
    logger = logging.getLogger('grab_classes')
    if mode == CONST.TRAIN or mode == CONST.VAL:
        classes = df_unique
        print(df_unique)
    else:

        # check if file is able to be opened
        try:
            f = open(filename, 'r')
        except Exception as e:
            logger.error(e)
            sys.exit()

        classes = parse_classes(filename)

    return sorted(classes)


def parse_classes(filename):
    """Parse MODE_data.info file"""
    lbs_all_classes = []
    with open(filename, 'r') as f:
        label_counts = f.readlines()
    label_counts = label_counts[:-1]
    for i in label_counts:
        class_counts = i.strip()
        class_counts = class_counts.split()
        class_name = ''
        for j in class_counts:
            if not j.isdigit():
                class_name += (' ' + j)
        class_name = class_name.strip()
        lbs_all_classes.append(class_name)
    return lbs_all_classes

def get_mapping_dict(mapping_csv=None, original_label_col='Original Label',
                     mapped_label_col='New Label'):
    logger = logging.getLogger(__name__)
    if not mapping_csv:
        mapping_csv = opt.hab_eval_mapping
        logger.debug(f'Classes file not given. Defaulting to {mapping_csv}')
    mapping_df = pd.read_csv(mapping_csv)
    mapping_dict = dict(zip(mapping_df[original_label_col], mapping_df[mapped_label_col]))
    return mapping_dict
