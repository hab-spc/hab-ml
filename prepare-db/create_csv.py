"""Create CSV file to hold image meta data

Main usages:
- [TEMP] Generate csv files of in situ images retrieved from the SPC database
- Generate csv files of in vitro images from the lab
- - - images given in year-month-day format, so need to process all images
at once

"""
# Standard dist imports
import glob
import os
import re

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
#TODO log dataset statistics from this
from data.d_utils import train_val_split, preprocess_dataframe

# Module level constants
DATA_DIR = '/data6/lekevin/hab-spc/phytoplankton-db'

def create_proro_csv(filter_word='proro', output_dir=DATA_DIR+'/csv/proro'):
    proro_dir = [f for f in os.listdir(DATA_DIR) if filter_word in f][::-1]

    proro_df = pd.DataFrame()

    for p_dir in proro_dir:
        data = glob.glob(os.path.join(DATA_DIR, p_dir))

        TRAINVAL = len(data) == 1

        if TRAINVAL:
            train_dir = data[0]
            if 'trainval' in train_dir:
                proro_types = glob.glob(train_dir+'/*')
            for p_type in proro_types:
                pd_dict = {}
                pd_dict['images'] = glob.glob(p_type+'/*')
                pd_dict['label'] = os.path.basename(p_type)
                proro_df = proro_df.append(pd.DataFrame(pd_dict))

            proro_df.to_csv(os.path.join(output_dir, 'proro_trainval_raw.csv'))
            proro_df = preprocess_dataframe(proro_df, proro=True,
                                            enable_uniform=True)

            # Train val split
            train_df, val_df = train_val_split(proro_df)

            # Save csv files
            train_df.to_csv(os.path.join(output_dir, 'proro_train.csv'))
            val_df.to_csv(os.path.join(output_dir, 'proro_val.csv'))

        else:
            pd_dict = {}
            pd_dict['images'] = data
            pd_dict['label'] = np.nan
            proro_df.append(pd.DataFrame(pd_dict))
    proro_df.to_csv(os.path.join(output_dir, 'proro_test.csv'))





create_proro_csv()