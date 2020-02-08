""" """
# Standard dist imports
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
# Third party imports
from PIL import Image
# Project level imports
from data.parse_data import DataParser
from data.label_encoder import HABLblEncoder

from utils.eval_utils import EvalMetrics
from utils.constants import Constants as CONST


class ReportGenerator(DataParser, EvalMetrics):
    def __init__(self, csv_fname=None, json_fname=None, model_path=None,
                 hab_eval=True, save=False):
        """Initializes ReportGenerator

        Args:
            csv_fname (str):
            json_fname (str):
            model_path (str):
            hab_eval (bool):
            save (bool):
        """
        if csv_fname:
            self.csv_fname = csv_fname
            self.csv_data = pd.read_csv(csv_fname)
            self.dataset = self.csv_data.copy()

        if json_fname:
            self.json_data = json.load(open(json_fname, 'rb'))
            self.dataset = pd.DataFrame(self.json_data['machine_labels'])
            self.dataset = self.dataset.rename({'gtruth': 'label'}, axis=1)

        if csv_fname and json_fname:
            self.dataset = self.merge_dataset(self.csv_data, self.json_data, save=save)

        if model_path:
            # Gets all of the dataset statistics from the trained model directory
            # and initializes them as attributes for usage throughout the script
            # No handling atm if attributes are used without the model_path given
            train_fname = os.path.join(model_path, 'train_data.info')
            val_fname = os.path.join(model_path, 'val_data.info')
            self.model_path = model_path
            self.class_list, \
            self.cls_count_train, \
            self.cls_count_val = self.get_dataset_statistics(train_fname, val_fname)
            self.cls2idx, self.idx2cls = self.set_encode_decode(self.class_list)
            self.le = HABLblEncoder(model_dir=model_path)

        # Initialize with prediction column to use given the hab_evaluation
        self.hab_eval = hab_eval
        self.pred_col = CONST.HAB_PRED if hab_eval else CONST.PRED

        # Get the hab species of interest for class filtering
        self.hab_species = open('/data6/phytoplankton-db/hab.txt', 'r').read().splitlines()

    def show_dataset_statistics(self):
        print('\n{0:*^80}'.format(' Dataset '))
        print('Idx. Cls\t\t\t\t\tTrainImageCount\t\tValImageCount')
        sum_train = sum(list(self.cls_count_train.values()))
        sum_val = sum(list(self.cls_count_val.values()))
        for idx, clss in enumerate(self.cls_count_train):
            print('{:2}. {:50} {:5}\t\t{:5}'.format(idx + 1, clss, self.cls_count_train[clss],
                                                    self.cls_count_val[clss]))
        print('Totals\nTrain: {} | Validation: {}'.format(sum_train, sum_val))
        print('Dataset location:\n{}\n{}')

    def show_testset_statistics(self):
        gt = self.get_gtruth()
        self._plot_dataset_distribution(data_dict=gt)

    def show_prediction_statistics(self):
        gt = self.get_gtruth()
        pr = self.get_predictions()
        pr = {k:v for (k,v) in pr.items() if v > 0}
        clsses = sorted(list(set(list(pr.keys())).union(list(gt.keys()))))
        print('\n{0:*^80}'.format(' Predictions '))
        print('Idx. Cls\t\t\t\t\t\tGtruth\t\tPredictions')
        for idx, clss in enumerate(clsses):
            print('{:2}. {:50} {:5}\t\t{:5}'.format(idx + 1, clss, gt[clss] if clss in gt else 0,
                                                    pr[clss] if clss in pr else 0))

    # TO BE DEPRECATED
    def class_mapping(self, class_name):
        # If there are mismatch class names between pred and gtruth, modify this
        # input and output are strings
        return class_name

    # It will get test acc and also return a list of idx of wrong predicted imgs
    def get_acc(self):
        c = 0
        wrong_idx = []
        for index, row in self.dataset.iterrows():
            a = row[CONST.LBL]
            a = self.class_mapping(a)
            b = row[self.pred_col]
            b = self.class_mapping(b)
            if a == b:
                c += 1
            else:
                wrong_idx.append(index)
        acc = c / self.dataset.shape[0]
        return [acc, wrong_idx]

    # It show a single img given image index
    def show_img(self, plot_idx, idx, original_img_flag=True, data_transform_1_flag=False, print_img_info=True):
        row = self.dataset.iloc[idx]
        img_path = row[CONST.IMG]

        pred = self.class_mapping(row[self.pred_col])
        gtruth = self.class_mapping(row[CONST.LBL])

        facecolor = 'red' if pred != gtruth else 'wheat'

        plot_idx.text(0.65, 0.85, 'Prediction: {}\nGtruth: {}'.format(pred, gtruth),
                         bbox=dict(facecolor=facecolor, alpha=0.75),
                         horizontalalignment='center', verticalalignment='bottom', transform=plot_idx.transAxes)

        # if print_img_info:
            # printmd('##### Image Path: ' + img_path)
            # printmd('##### Gtruth: ' + self.class_mapping(self.classes[row['gtruth']]))
            # printmd('##### Prediction: ' + self.class_mapping(self.classes[row['pred']]))

        input_size = rescale_size = 224

        data_transform_1 = {
            'train': transforms.Compose([transforms.Resize(rescale_size),
                                         transforms.CenterCrop(input_size),
                                         transforms.RandomAffine(360, translate=(0.1, 0.1), scale=None, shear=None,
                                                                 resample=False, fillcolor=0),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ]),
            'val': transforms.Compose([transforms.Resize(rescale_size),
                                       transforms.CenterCrop(input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ]),
            'deploy': transforms.Compose([transforms.Resize(rescale_size),
                                          transforms.CenterCrop(input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                          ])
        }
        if original_img_flag:
            img_ori = Image.open(img_path)
            plot_idx.imshow(img_ori)
        if data_transform_1_flag:
            img_1 = Image.open(img_path)
            img_size = img_1.size
            img_padding = [0, 0]
            if img_size[0] > img_size[1]:
                img_padding[1] = int((img_size[0] - img_size[1]) / 2)
            else:
                img_padding[0] = int((img_size[1] - img_size[0]) / 2)
            img_padding = tuple(img_padding)
            img_1 = transforms.Pad(img_padding, fill=0, padding_mode='constant')(img_1)
            img_1 = data_transform_1['deploy'](img_1)
            img_1 = img_1.data.numpy()
            img_1 = np.transpose(img_1, (1, 2, 0))
            plot_idx.imshow(img_1)

    # Get a list of indexes of which matches certain value in certrain column.
    # Can be used to combine with show_multi_img
    # EX: show_multi_img( get_column_value_idx('detritus','gtruth') )
    def get_column_value_idx(self, value, column_name):
        return self.dataset.index[self.dataset[column_name] == value].tolist()

    # Show multiple images given img index in self.data
    def show_multi_img(self, idx_list, NUM_COLS=8, MAX_ROWS=5):
        num_rows = int(max(len(idx_list) / NUM_COLS + 1, 2))
        
        row_max_idx = min(num_rows, MAX_ROWS)
        fig, axarr = plt.subplots(row_max_idx, NUM_COLS, figsize=(30, 10))
        for i_ax, img in enumerate(idx_list):
            if int(i_ax / NUM_COLS) >= row_max_idx:
                break
            plot_idx = axarr[int(i_ax / NUM_COLS), i_ax % NUM_COLS]
            self.show_img(plot_idx, img)
            plot_idx.set_axis_off()
        plt.show()

    def show_misclassifications(self):
        """Show misclassifications"""
        for cls in sorted(self.dataset[CONST.LBL].unique()):
            print('\n{0:*^80}'.format(' {} '.format(cls)))
            self.show_multi_img(self.get_column_value_idx(cls, CONST.LBL))

    #Show Test Confusion Matrix
    def show_confusion_matrix(self):
        self.classes = self.le.hab_classes if self.hab_eval else self.class_list
        self.num_class = len(self.classes)
        self.gtruth = self.dataset[CONST.LBL].map(self.le.habcls2idx).tolist()
        self.predictions = self.dataset[self.pred_col].map(self.le.habcls2idx).tolist()
        self.compute_cm(plot=True, save=False)

    # Show Val Confusion Matrix
    def show_val_confusion_matrix(self):
        img_path = os.path.join(self.model_path, 'figs/train_confusion.png')
        img = Image.open(img_path)
        plt.figure(figsize=(65, 65))
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off')
        plt.show()

    # Show loss graph
    def show_loss_graph(self):
        img_path = os.path.join(self.model_path, 'figs/train_val_loss.png')
        img = Image.open(img_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.show()

    # Show acc graph
    def show_acc_graph(self):
        img_path = os.path.join(self.model_path, 'figs/train_val_accuracy.png')
        img = Image.open(img_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.show()

