""" """
import json
# Standard dist imports
import logging
import os

import numpy as np
# Third party imports
import pandas as pd

# Project level imports
from utils.constants import Constants as CONST

class DataParser(object):
    """Parse data"""

    def __init__(self, csv_fname=None, json_fname=None, model_path=None,
                 hab_eval=True, save=False):
        """ Initializes SPCParsing instance to extract data from dataset

        Args:
            csv_fname (str): Abs path to the csv file (dataset)
            json_fname (str): Abs path to the json file containing predictions
            classes (str): Abs path to the training.log that contains the classes
            save (bool): Flag to save the merged dataset
        """
        if csv_fname:
            self.csv_fname = csv_fname
            self.csv_data = self.read_csv_dataset(csv_fname)
            self.dataset = self.csv_data.copy()

        if json_fname:
            self.json_data = json.load(open(json_fname, 'rb'))
            self.dataset = pd.DataFrame(self.json_data['machine_labels'])
            self.dataset = self.dataset.rename({'gtruth': 'label'}, axis=1)

        if csv_fname and json_fname:
            self.dataset = self.merge_dataset(self.csv_data, self.json_data, ave=save)

        if model_path:
            train_fname = os.path.join(model_path, 'train_data.info')
            val_fname = os.path.join(model_path, 'val_data.info')
            self.class_list, \
            self.cls_count_train, \
            self.cls_count_val = self.get_dataset_statistics(train_fname, val_fname)

            self.cls2idx, self.idx2cls = self.set_encode_decode(self.class_list)

        self.pred_col = CONST.HAB_PRED if hab_eval else CONST.PRED

        # Get the hab species of interest for class filtering
        self.hab_species = open('/data6/phytoplankton-db/hab.txt', 'r').read().splitlines()

    def read_csv_dataset(self, csv_file, verbose=False):
        df = pd.read_csv(csv_file)
        if verbose:
            print('\n{0:*^80}'.format(' Reading in the dataset '))
            print("\nit has {0} rows and {1} columns".format(*df.shape))
            print('\n{0:*^80}\n'.format(' It has the following columns '))
            print(df.info())
            print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
            print(df.head())
        return df

    def merge_dataset(self, csv_data, json_data, save=False):
        # Drop outdated `label` column (used as gtruth in machine learning exp)
        label_col = 'label'
        meta_df = csv_data.copy()
        meta_df = meta_df.rename({'timestamp': 'image_timestamp'})

        # Preprocess prediction json
        pred_df = pd.DataFrame(json_data['machine_labels'])
        pred_df = pred_df.rename({'gtruth': label_col}, axis=1)
        pred_df['image_id'] = pred_df['image_id'].apply(DataParser.extract_img_id)

        # Merge based off image_id
        merged = meta_df.merge(pred_df, on='image_id')

        # Overwrite previous csv file with new gtruth
        if save:
            csv_fname = self.csv_fname.split('.')[0] + '-predictions.csv'
            print(f'Saved as {csv_fname}')
            merged.to_csv(csv_fname, index=False)
        return merged

    #==== BEGIN: GETTER CALLS ===#
    def get_gtruth(self, gtruth_col=CONST.LBL, verbose=False, decode=False):
        """Get the gtruth distributions"""
        gtruth = self.dataset[gtruth_col]
        if self.idx2cls and decode:
            gtruth = self.dataset[gtruth_col].map(self.idx2cls)

        if verbose:
            print(gtruth.value_counts())
        self.gtruth = gtruth.value_counts().to_dict()
        return self.gtruth

    def get_predictions(self, pred_col=None, verbose=False, decode=False):
        """Get the prediction distribution"""
        if pred_col is None:
            pred_col = self.pred_col

        pred = self.dataset[pred_col]
        if self.idx2cls and decode:
            pred = self.dataset[pred_col].map(self.idx2cls)

        if verbose:
            print(pred.value_counts())
        self.pred = pred.value_counts().to_dict()
        for idx,cls in enumerate(self.class_list):
            if cls not in self.pred and decode:
                self.pred[cls] = 0
                continue
            elif idx not in self.pred:
                self.pred[idx] = 0
                continue

        return self.pred

    def get_dataset_statistics(self, train_fname, val_fname):
        class_list, images_per_class_train = DataParser._parse_info_file(train_fname)
        _, images_per_class_val = DataParser._parse_info_file(val_fname)
        return class_list, images_per_class_train, images_per_class_val

    def get_classes(self, data):
        df = data.copy()
        classes = df[CONST.LBL].unique()
        num_class = len(classes)
        return classes, num_class

    def set_encode_decode(self, class_list):
        """Set class2idx, idx2class encoding/decoding dictionaries"""
        cls2idx = {i: idx for idx, i in enumerate(sorted(class_list))}
        idx2cls = {idx: i for idx, i in enumerate(sorted(class_list))}
        return cls2idx, idx2cls

    def get_ROI_counts(self, data, date_col='image_date', gtruth=False, pred=False, verbose=False):
        """ Given a particular date column retrieve all ROIs within the time range

        Region of Interest (RoI) is considered an image

        ROI_counts['date'] = ['ROI_count', 'gtruth_dist', 'pred_dist']

        Args:
            data (pd.DataFrame): dataframe to run RoI count

        Returns:

        """
        df = data.copy()
        grouped_dates = df.groupby(date_col)
        ROI_counts = {}
        for idx, date_df in grouped_dates:
            temp = {}
            num_ROIs = date_df.shape[0]

            temp['ROI_count'] = num_ROIs
            if gtruth:
                temp['gtruth_dist'] = df['label'].map(self.idx2cls).value_counts().to_dict()
            if pred:
                temp['pred_dist'] = df['pred'].map(self.idx2cls).value_counts().to_dict()
            temp['start_time'] = date_df['start_time'].iloc[0]
            temp['end_time'] = date_df['end_time'].iloc[0]
            temp['min_len'] = date_df['min_len'].iloc[0]
            temp['max_len'] = date_df['max_len'].iloc[0]
            temp['cam'] = date_df['cam'].iloc[0]

            ROI_counts[str(idx)] = temp

        return ROI_counts

    #==== END: GETTER CALLS ===#


    @staticmethod
    def extract_img_id(x):
        return os.path.basename(x).split('.')[0] + '.tif'

    @staticmethod
    def _parse_info_file(filename=None):
        """Parse MODE_data.info file"""
        lbs_all_classes = []
        img_per_classes = []

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
                else:
                    img_per_classes.append(int(j))
            class_name = class_name.strip()
            lbs_all_classes.append(class_name)

        class_dct = {}
        c = 0
        for i in lbs_all_classes:
            class_dct[c] = i
            c += 1

        img_per_class = {}
        c = 0
        for i in lbs_all_classes:
            img_per_class[i] = img_per_classes[c]
            c += 1
        return lbs_all_classes, img_per_class
