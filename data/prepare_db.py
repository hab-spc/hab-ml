"""Prepare training/prediction dbs for Model

Main script downloads images given the time query and preprocesses it into a
dataframe format for 'SPCHabDataset' initialization.

#TODO give option to train_val_split, if database preparation is for training
- i.e. take in option to train or predict on the data


"""
# Standard dist imports
from datetime import datetime
import glob
import logging
import os
import sys
sys.path.extend(['/data6/lekevin/hab-master',
                 '/data6/lekevin/hab-master/hab-spc'])

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
#TODO log dataset statistics from this
from utils.logger import Logger
from utils.constants import SPCData as sd
from data.d_utils import train_val_split, preprocess_dataframe
from spc.spcserver import SPCServer
from spc.spctransformer import SPCDataTransformer
from spc.spc_go import parse_cmds

# Module level constants
DEBUG = False

def prepare_db(data=None, image_dir=None, csv_file=None, save=False,
               args=None):
    """
    #TODO use this as test data for now. CSV files will have gtruth
    Right now gtruth includes items like "False Prorocentrum", which should 
    be "Non-Prorocentrum", but leaving that up to you to modify.
    
    After getting some results, use that data to retrain the classifier again.
    #TODO create retraining-pipeline
    """

    if not data and isinstance(data, pd.DataFrame):
        if os.path.exists(image_dir):
            tf_df = SPCDataTransformer(data=data).transform(image_dir)

    else:
        spc = SPCServer()
        spc.retrieve(textfile=args.search_param_file,
                     output_dir=args.image_output_path,
                     output_csv_filename=args.meta_output_path,
                     download=args.download)

        # Read in resulting csv file
        csv_file = args.meta_output_path
        df = pd.read_csv(csv_file)
        tf_df = SPCDataTransformer(data=df).transform(args.image_output_path)

    # Resave with added transformations
    if save:
        print('Dataset prepared. Saved as {}'.format(csv_file))
        tf_df.to_csv(csv_file, index=False)

    return tf_df

def create_lab_csv(data_root):
    import glob
    import pandas as pd
    data_root += '_static_html'
    images_dir = glob.glob(os.path.join(data_root, 'images', '*'))
    images = []
    for d in images_dir:
        images.extend(glob.glob(os.path.join(d, '*-.jpeg')))
    img_id = [os.path.basename(i).replace('.jpeg', '.tif') for i in images]
    df = pd.DataFrame({sd.IMG: images, sd.ID:img_id, sd.LBL: 0, sd.USR_LBL:0})
    csv_fname = os.path.join(data_root, 'meta.csv')
    df.to_csv(csv_fname, index=False)
    return csv_fname


def create_proro_csv(filter_word='proro', data_dir=None):
    """Create prorocentrum training/eval dataset ONLY for PRORO"""
    output_dir = os.path.join(data_dir, 'csv/proro')
    Logger(os.path.join(output_dir, 'proro_csv.log'), logging.DEBUG,
           log2file=False)
    Logger.section_break('Create Proro-CSV')
    logger = logging.getLogger('create-csv')

    proro_dir = [f for f in os.listdir(data_dir) if filter_word in f][::-1]

    proro_df = pd.DataFrame()

    for p_dir in proro_dir:
        data_dir_ = os.path.join(data_dir, p_dir)

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

            def create_img_id(x):
                return os.path.basename(x).split('.')[0] + '.tif'

            # Create image id
            proro_df['image_id'] = proro_df['images'].apply(create_img_id)

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

if __name__ == '__main__':
    if DEBUG:
        from argparse import Namespace

        working_dir = '/data6/lekevin/hab-master/hab-spc/data'
        textfile = os.path.join(working_dir, 'experiments/time_period.txt')
        output_dir = os.path.join(working_dir, 'experiments/images')

        hour = False
        save_fmt = '%Y-%m-%d_%H:%M:%S' if hour else '%Y-%m-%d'
        mode_fmt = 'test'

        output_csv_file = os.path.join(
            working_dir, 'experiments/proro_{}_{}.csv'.format(
                mode_fmt, datetime.today().strftime(save_fmt)))

        args = Namespace(search_param_file=textfile,
                         image_output_path=output_dir,
                         meta_output_path=output_csv_file,
                         download=False)
        prepare_db(args=args)

    else:
        prepare_db(args=parse_cmds())

