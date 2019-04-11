"""Create CSV file to hold image meta data

Main usages:
- [TEMP] Generate csv files of in situ images retrieved from the SPC database
- Generate csv files of in vitro images from the lab
- - - images given in year-month-day format, so need to process all images
at once

"""
# Standard dist imports
import glob
import logging
import os
import re

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
#TODO log dataset statistics from this
from utils.logger import Logger
from data.d_utils import train_val_split, preprocess_dataframe

# Module level constants
DATA_DIR = '/data6/lekevin/hab-spc/phytoplankton-db'

def create_proro_csv(filter_word='proro', output_dir=DATA_DIR+'/csv/proro'):
    Logger(os.path.join(output_dir, 'proro_csv.log'), logging.DEBUG,
           log2file=False)
    Logger.section_break('Create Proro-CSV')
    logger = logging.getLogger('create-csv')

    proro_dir = [f for f in os.listdir(DATA_DIR) if filter_word in f][::-1]

    proro_df = pd.DataFrame()

    for p_dir in proro_dir:
        data_dir_ = os.path.join(DATA_DIR, p_dir)

        TRAINVAL = 'trainval' in data_dir_
        logger.info('Creating csv for {}'.format(p_dir))

        if TRAINVAL:
            # Prepare training and validation set
            proro_types = glob.glob(data_dir_ + '/*')

            # Parse images and labels from each annotation set into dataframe
            for p_type in proro_types:
                pd_dict = {}
                pd_dict['images'] = glob.glob(p_type + '/*')
                pd_dict['label'] = os.path.basename(p_type)
                proro_df = proro_df.append(pd.DataFrame(pd_dict))

            # Save copy of raw data before preprocessing
            fn = 'proro_trainval_raw.csv'
            logger.debug('Saving raw version of dataset as {}'.format(fn))
            proro_df.to_csv(os.path.join(output_dir, fn))
            proro_df = preprocess_dataframe(proro_df, logger, proro=True,
                                            enable_uniform=True)

            # Train val split
            train_df, val_df = train_val_split(proro_df)

            # Save csv files
            train_fn = os.path.join(output_dir, 'proro_train.csv')
            val_fn = os.path.join(output_dir, 'proro_val.csv')
            train_df.to_csv(train_fn, index=False)
            val_df.to_csv(val_fn, index=False)
            logger.info('Saved as:\n\t{}\n\t{}\n'.format(train_fn, val_fn))

        else:
            # Prepare test set of unknown labels
            data = glob.glob(data_dir_ + '/*')
            logger.info('Total unlabeled images: {}'.format(len(data)))

            # Set up test dataframe
            pd_dict = {}
            pd_dict['images'] = data  # Parsed image filenames
            pd_dict['label'] = np.nan  # Unknown label
            proro_df = pd.DataFrame(pd_dict)
            test_fn = os.path.join(output_dir, 'proro_test.csv')
            proro_df.to_csv(test_fn, index=False)
            logger.info('Saved as:\n\t{}'.format(test_fn))


# create_proro_csv()

def create_density_csv():
    data_dir = '/data6/lekevin/hab-spc'
    micro_Data = pd.read_csv(os.path.join(data_dir, "rawdata/Micro_data.csv"))
    #TODO prediction counts should come from proro_<PARTITION>.csv files
    image_Data = pd.read_csv(os.path.join(data_dir, "rawdata/Image_data.csv"),
                             skiprows=[1, 2, 3])

    # Process Image_data to merge with microscopy data
    image_Data = image_Data[['Start Date & Time', 'End Date & Time',
                             '# Non-Proro detected', '# Proro detected']]
    image_Data['Date'] = image_Data['Start Date & Time'].str.split(' ').str[0]
    image_Data['Date'] = pd.to_datetime(image_Data['Date'])
    image_Data['Date'] = image_Data['Date'].dt.strftime('%Y-%m-%d')

    # Process Microscopy_data
    CSV_COLUMNS = 'Date,Prorocentrum micans (Cells/L),Total Phytoplankton (Cells/L)'
    micro_Data = micro_Data.rename(columns={'Date\n\nmm/dd/yy': "Date"}, index=str)
    micro_Data['Date'] = pd.to_datetime(micro_Data['Date']).dt.strftime('%Y-%m-%d')
    micro_Data = micro_Data[CSV_COLUMNS.split(',')]

    # Merge two data types
    density_Data = micro_Data.merge(image_Data, on='Date')

    # Rename columns for simplicity
    rename_dict = {'Prorocentrum micans (Cells/L)': 'micro_proro',
                   'Total Phytoplankton (Cells/L)': 'micro_total-phyto',
                   'Start Date & Time': 'image_start-datetime',
                   'End Date & Time': 'image_end-datetime',
                   '# Non-Proro detected': 'image_non-proro',
                   '# Proro detected': 'image_proro'}
    density_Data = density_Data.rename(columns=rename_dict, index=str)

    # Compute total phytoplankton
    density_Data['image_total-phyto'] = density_Data['image_non-proro'] + \
                                        density_Data['image_proro']

    # Save as raw data
    fname = os.path.join(data_dir, 'rawdata/Density_data.csv')
    density_Data.to_csv(fname, index=False)



create_density_csv()


