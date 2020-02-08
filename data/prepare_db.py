"""Prepare training/prediction dbs for Model

# download new hab_in_situ data
# model report

"""
import argparse
import os
import random
import sys
from pathlib import Path

# Standard dist imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
from data.d_utils import read_csv_dataset
from utils.config import opt
from utils.constants import SPCConstants as SPC_CONST

# Module level constants
DEBUG = False
INSITU = 'hab_in_situ'
OTHER_LBL = 'other'


class DatasetGenerator(object):
    def __init__(self, csv_file, data_dir, labels=None, dates=None, pier=None, lab=None):
        # Initialize main data directory to save data
        self.data_dir = data_dir

        # Read in the dataset from the csv file
        try:
            self.data = read_csv_dataset(csv_file)
        except:
            raise FileNotFoundError('File does not exist: {}'.format(csv_file))

        if pier:
            self.data = self.add(pier)

        if lab:
            self.data = self.add(lab)

        # TODO inteligently sample added data

        # preprocess insitu datasets for empty, double labels
        # if not workshop, but insitu
        print('Cleaning & filtering dataframe\n')
        parts = os.path.basename(csv_file)
        if INSITU in parts and (not parts.split('.')[0].endswith('workshop2019')):
            self.data = DatasetGenerator.clean_raw_pier_labels(data=self.data)

        # Given a labels txt file of the selected labels
        # filter the dataset
        if labels:
            self.data = self.filter_labels(labels)

        # Given a dates txt file of the selected dates
        # filter the dataset
        if dates:
            self.data = self.filter_dates(dates)

    def add(self, data_fname):
        data = read_csv_dataset(data_fname)
        # read in hab species, then filter labels based off that
        #todo filtering labels
        classes = {1: 'Akashiwo',
                   2: 'Ceratium falcatiforme or fusus',
                   3: 'Ceratium furca',
                   6: 'Chattonella',
                   8: 'Cochlodinium',
                   11: 'Gyrodinium',
                   12: 'Lingulodinium polyedra',
                   15: 'Prorocentrum michans',
                   17: 'Pseudo-nitzschia chain'}
        data['label'] = data['label'].map(classes)
        data = data[data['label'].isin(classes.values())].reset_index(drop=True)

        df = self.data.copy()
        return pd.concat([df, data])

    @staticmethod
    def clean_raw_pier_labels(data, annotated_label_col=SPC_CONST.USR_LBL,
                              gtruth_label_col=SPC_CONST.LBL, drop=False):
        """Clean raw data (i.e. parse labels"""
        df = data.copy()

        def map_labels(x):
            label = eval(x)
            empty_label = len(label) == 0
            multiple_labels = len(label) >= 2

            # If we run into an empty label,
            # right now we want to assume that as the
            # common class "Other" until we can get someone
            # to identify the class
            if empty_label:
                assigned_label = OTHER_LBL

            # If we run into multiple labels
            # ideally we want to use the labels from the true expert if available
            # otherwise, we just use the trained annotators. For now since we dont
            # have those labels, we'll put a placeholder for randomly sampling it
            elif multiple_labels:
                if OTHER_LBL in label:
                    label.pop(label.index(OTHER_LBL))
                assigned_label = str(random.sample(label, k=1)[0])

            # Default case is to return the single label assigned
            else:
                assigned_label = str(label[0])
            return assigned_label

        # Map labels given label parsing scheme
        df[gtruth_label_col] = df[annotated_label_col].apply(map_labels)
        if drop:
            df = df.drop(annotated_label_col, axis=1)

        return df
        # map empty labels # map double labels

    def filter_dates(self, dates_txt_file, date_col='image_date'):
        """Dates ~ naming convention for training & test sets"""
        dates = self._read_dates(dates_txt_file)

        df = self.data.copy()
        return df[df[date_col].isin(dates)].reset_index(drop=True)

    def filter_labels(self, labels_txt_file, data=None, label_col=SPC_CONST.LBL):
        """Select classes given a label text file"""
        # Read labels
        labels = self._read_labels(labels_txt_file)

        if data:
            df = data.copy()
        else:
            df = self.data.copy()

        df = df[df[label_col].isin(labels.values())].reset_index(drop=True)
        return df

    def train_val_split(self):
        data_dir = self.data_dir
        df = self.data.copy()

        # randomly shuffle the rows in df
        print('Randomly Shuffle the Dataset')
        df.sample(frac=1)

        freq = df[SPC_CONST.LBL].value_counts()
        print('\n{0:*^80}'.format(' Image/Class Statistics '))
        print('Image/Class Frequency:\n{}\n{}\n'.format('-' * 30, freq))
        print('Current list of classes ({}): \n{}\n'.format(len(freq), sorted(df[SPC_CONST.LBL].unique())))

        stop = False
        print('\n{0:*^80}'.format(' Begin Dataset Partioning '))
        while stop == False:
            # initialize data structure for specifying number of images
            # per class within training set. Remaining images are automatically
            # allocated for the validation set
            train_dict = {}
            classes = df[SPC_CONST.LBL].unique()

            # begin partitioning here
            opt_partition = input("Current partioning/sampling options\n{}\n"
                                  "\tType 1 to partition each class by a train_val ratio [Default]. \n"
                                  "\tType 2 to partition each class by a user_selected sampling.\n".format('-' * 40))

            if opt_partition == '1':
                nums = self.train_val_ratio(classes)

            elif opt_partition == '2':
                nums = self.user_selected_sampling(classes)
            else:
                print('Option not given!')
                stop = False
                continue

            # FLAGS for establishing training dictionaries
            train_val_ratio_flag = len(nums) == 1
            user_input_sampling_flag = len(nums) == len(classes)

            if train_val_ratio_flag:
                train_dict = {i: int(nums[0]) for i in classes}

            elif user_input_sampling_flag:
                train_dict = {classes[i]: int(nums[i]) for i in range(len(classes))}

            else:
                print('Class number not match !')
                stop = False
                continue

            # Request user input to verify dataset partitioning
            # then break the loop if `y`
            print('Numbers of images selected for each class for training: ')
            print(train_dict)
            y_n = input('Are you sure these are the number you want? (y/n): (ex. y) \n')
            stop = True if y_n == 'y' else False

        # Create train and val df. train_df: df of train.csv; df: df of val.csv
        train_df = pd.DataFrame()
        # shuffle dataset
        df = df.iloc[np.random.permutation(len(df))]
        df = df.reset_index(drop=True)
        for i in classes:
            # given the number of images per class within training dictionary
            # sample without replacement that given amount
            temp = df.loc[df[SPC_CONST.LBL] == i][:train_dict[i]]
            train_df = train_df.append(temp)
            # drop the training samples from the original dataframe
            # to use the remainning for validation
            temp = temp.index
            df = df.drop(temp)
        val_df = df.reset_index(drop=True)

        # === Labels Processing ===#
        print('Start Labels Processing')
        # Combined Straight + Curved diatom chains into diatom chain
        print('Combine Curved and Straight diatom chain into diatom chain')
        train_df.loc[train_df[SPC_CONST.LBL].isin(
            ['Curved diatom chain', 'Straight diatom chains']), SPC_CONST.LBL] = 'diatom chain'
        val_df.loc[val_df[SPC_CONST.LBL].isin(
            ['Curved diatom chain', 'Straight diatom chains']), SPC_CONST.LBL] = 'diatom chain'

        # === Dataset stats logging & saving ===#
        print('Train and Val Dataframe contructed')
        print('Train Dataframe Class Counts table')
        print(train_df['label'].value_counts())
        print('VAL Dataframe Class Counts table')
        print(val_df['label'].value_counts())

        # Store csv and info.txt
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        train_path = os.path.join(data_dir, 'train.csv')
        train_df.to_csv(train_path, index=False)
        print('Train Dataframe is stored to ' + train_path)

        val_path = os.path.join(data_dir, 'val.csv')
        val_df.to_csv(val_path)
        print('Validation Dataframe is stored to ' + val_path)

        info_path = os.path.join(data_dir, 'info.txt')
        print('Info is stored to ' + info_path)
        f = open(info_path, 'w')
        sys.stdout = f
        print('Train and Test Dataframe contructed')
        print('Train Dataframe Class Counts table')
        print(train_df['label'].value_counts())
        print('VAL Dataframe Class Counts table')
        print(val_df['label'].value_counts())
        f.close()

    def save(self):
        """Save the dataset after preparation"""
        df = self.data.copy()
        csv_filename = os.path.join(self.data_dir, os.path.basename(self.data_dir) + '.csv')
        print('Dataset saved as {}'.format(csv_filename))
        df.to_csv(csv_filename, index=False)

    def train_val_ratio(self, classes):
        """Parition each class by a percentage"""
        df = self.data.copy()
        nums = []
        percent = input(
            """Select percentages of images for each class to be in train.csv? \n 
            Ex. 0.8 ==> class1: 80% train| 20% val; class2: 80% train| 20% val, ... \n""")
        percent = float(percent)
        for i in classes:
            # Retrieve image counts grouped by each class and
            # compute the data by the percent
            temp = df.loc[df[SPC_CONST.LBL] == i].shape[0]
            num = int(temp * percent)
            nums.append(num)
        return nums

    def user_selected_sampling(self, classes):
        """Partition each class by sampling"""
        print('Enter number of images per class for training.\n')
        print(sorted(classes))
        nums = input(
            """Ex. `20,40,1,2,3,6` --> results in 'Prorocentrum micans': 20, 'Lingulodinium polyedra': 40, 
            Akashiwo': 1, 'Gyrodinium': 2, 'Cochlodinium': 3, 'Chattonella': 6}\n""")
        nums = nums.replace(" ", "")
        nums = nums.split(',')
        nums = [x for x in nums if x]
        return nums

    def _read_dates(self, dates_txt_file):
        """Read dates text file"""
        with open(dates_txt_file, 'r') as f:
            dates = set([line.strip() for line in f])
        f.close()
        return dates

    def _read_labels(self, labels_txt_file):
        """Read labels text file"""
        with open(labels_txt_file, 'r') as f:
            labels = {int(k): v for line in f for (k, v) in (line.strip().split(None, 1),)}
        f.close()
        return labels


def prepare_db(split, csv_file_path, data_dir,
               labels_txt=None, dates=None, add_pier=None, add_lab=None):
    # Select subset of the dataset according to the dates and classes
    # if given dates and classes txt files
    gen = DatasetGenerator(csv_file_path, data_dir, labels_txt, dates, add_pier, add_lab)

    # split the dataset if specified
    if split:
        gen.train_val_split()
    # if script is only used for preparation,
    # then save the dataset as default
    else:
        gen.save()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser('Prepare dataset')
    parser.add_argument('--train_val_split', '-tv', action='store_true', help='Flag for train_val_split')
    parser.add_argument('--csv', default=None, type=str, help='Absolute csv filepath (raw)')
    parser.add_argument('--labels', default=None, type=str, help='Absolute labels filepath')
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Relative data directory to DB')
    parser.add_argument('--dates', default=None, type=str, help='Absolute dates filepath')
    parser.add_argument('--add_pier', default=None, type=str, help='Absolute csv filepath (pier)')
    parser.add_argument('--add_lab', default=None, type=str, help='Absolute csv filepath (lab)')
    parser.add_argument('--reformat_db', action='store_true', help='Flag for reformatting dataset')
    args = parser.parse_args()

    # Prepare the dataset for training and validation
    prepare_db(split=args.train_val_split, csv_file_path=args.csv,
               data_dir=args.data_dir, labels_txt=args.labels, dates=args.dates,
               add_pier=args.add_pier, add_lab=args.add_lab)
