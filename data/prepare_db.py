"""Prepare training/prediction dbs for Model

Main script downloads images given the time query and preprocesses it into a
dataframe format for 'SPCHabDataset' initialization.

#TODO give option to train_val_split, if database preparation is for training
- i.e. take in option to train or predict on the data


"""
import os
import sys
# Standard dist imports
from datetime import datetime

sys.path.extend(['/data6/lekevin/hab-master',
                 '/data6/lekevin/hab-master/hab-rnd/hab-ml',])

# Third party imports
import pandas as pd

# Project level imports
#TODO log dataset statistics from this
from utils.constants import SPCConstants as SPC_CONST
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

    if (data is not None) and isinstance(data, pd.DataFrame):
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
    images_dir = glob.glob(os.path.join(data_root, 'static/images', '*'))
    images = []
    for d in images_dir:
        images.extend(glob.glob(os.path.join(d, '*-.jpeg')))
    img_id = [os.path.basename(i).replace('.jpeg', '.tif') for i in images]
    df = pd.DataFrame({SPC_CONST.IMG: images, SPC_CONST.ID: img_id, SPC_CONST.LBL: 0, SPC_CONST.USR_LBL: 0})
    csv_fname = os.path.join(data_root, 'meta.csv')
    df.to_csv(csv_fname, index=False)
    return csv_fname

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

