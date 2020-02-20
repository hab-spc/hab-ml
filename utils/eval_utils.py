from collections import OrderedDict
import itertools
import random
import logging
import math
import os

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import torch.nn as nn

from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

from utils.config import opt
from utils.constants import Constants as CONST
from data.label_encoder import HABLblEncoder

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.std = 0.

    def update(self, val, n=1):
        val = val.astype(float) if isinstance(val, np.ndarray) else float(val)
        self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = np.std(self.data)

def get_meter(meters=['batch_time', 'loss', 'acc']):
    return {meter_type: AverageMeter() for meter_type in meters}

def accuracy(predictions, targets, axis=1):
    batch_size = predictions.size(0)
    predictions = predictions.max(axis)[1].type_as(targets)
    hits = predictions.eq(targets)
    acc = 100. * hits.sum().float() / float(batch_size)
    return acc

class EvalMetrics(object):
    """ Evaluation metrics object to compute and plot basic metrics"""
    def __init__(self, classes, predictions=[], gtruth=[], ids=[], model_dir=''):
        """Initializes EvalMetrics

        Args:
            classes (list): List of unique classes
            predictions (list): List of predictions in the form of indices or strings
            gtruth (list): List of gtruth in the form of indices or strings
            ids (list): List of image identifiers
            model_dir (str): Abs path to the model directory for saving results
        """
        # Initialize passed arguments
        self.num_class = len(classes)
        self.classes = classes
        self.predictions = predictions
        self.gtruth = gtruth
        self.ids = ids

        # Initialize basename of the file as `pre` and create figs directory
        self.pre = ''
        self.results_dir = os.path.join(model_dir, 'figs')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Initialization for probabilities
        self.probabilities = []
        self.softmax = nn.Softmax(dim=1)

        # Initialize encoder for transforming labels into HAB labels
        self.le = HABLblEncoder()

        # Initialize set color for each class when plotting
        self.colors = None

        self.logger = logging.getLogger(__name__)

    def update(self, predictions, targets, ids, axis=1):
        """Update predictions and gtruth that are retrieved from training/eval loop

        Args:
            predictions (cuda): Batched predictions
            targets (cuda): Batched gtruth labels
            ids (cuda): Batched image identifiers
            axis (int): Axis to max predictions by

        Returns:

        """
        probs = self.softmax(predictions).detach().data.cpu().numpy()
        self.probabilities.extend(probs)
        
        predictions = predictions.detach().data.cpu().numpy()
        predictions = predictions.argmax(axis).astype(int)
        self.predictions.extend(predictions)

        targets = targets.detach().data.cpu().numpy().astype(int)
        self.gtruth.extend(targets)

        ids = ids
        self.ids.extend(ids)
    
    def compute_hab_acc(self):
        """ Compute HAB Accuracy

        Assumes that the classes are not already mapped to HAB class indices and
        transforms them into indices. This also initializes the gtruth, predictions,
        etc. for the rest of the evaluation processing in a hab_eval form.

        Returns:

        """
        self.num_class = 10
        self.classes = self.le.hab_classes
        self.gtruth = self.le.hab_transform2idx(self.gtruth)
        self.predictions = self.le.hab_transform2idx(self.predictions)

        c=0
        for t, p in zip(self.gtruth, self.predictions):
            if t==p:
                c+=1
        return c/len(self.gtruth)*100

    def compute_cm(self, plot=False, save=True):
        # Create array for confusion matrix with dimensions based on number of classes
        confusion_matrix_rawcount = np.zeros((self.num_class, self.num_class))
        class_count = np.zeros(
            (self.num_class, 1))  # 1st col represents number of images per class

        # Create confusion matrix
        for t, p in zip(self.gtruth, self.predictions):
            class_count[t, 0] += 1
            confusion_matrix_rawcount[t, p] += 1
        confusion_matrix_rate = np.zeros((self.num_class, self.num_class))
        for i in range(self.num_class):
            confusion_matrix_rate[i, :] = (confusion_matrix_rawcount[i, :]) / \
                                                  class_count[i, 0] * 100
            
        confusion_matrix_rate = np.around(confusion_matrix_rate, decimals=4)

        if plot:
            self._plot_confusion_matrix(confusion_matrix_rate, save=save)
        return confusion_matrix_rate, np.nanmean(np.diag(confusion_matrix_rate))

    def compute_roc_auc_score(self, plot=True):
        from scipy import interp

        def check_ohe_encoded(x):
            if isinstance(x, (np.ndarray, np.generic)):
                if x.shape[1] == self.num_class:
                    return False
            return True

        if check_ohe_encoded(self.gtruth) or check_ohe_encoded(self.predictions):
            gtruth, predictions = binarize_scores(self.gtruth,
                                                  self.predictions,
                                                  self.num_class)

        fpr, tpr = {}, {}
        roc_auc = {}
        for i in range(self.num_class):
            fpr[i], tpr[i], _ = metrics.roc_curve(gtruth[:, i], predictions[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(gtruth.ravel(),
                                                          predictions.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_class)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_class):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.num_class

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        if plot:
            self._plot_roc_curve(fpr, tpr, roc_auc)
        return fpr, tpr, roc_auc

    def compute_precision_recall_ap_score(self, plot=False):
        def check_ohe_encoded(x):
            if isinstance(x, (np.ndarray, np.generic)):
                if x.shape[1] == self.num_class:
                    return False
            return True

        if check_ohe_encoded(self.gtruth) or check_ohe_encoded(self.predictions):
            gtruth, predictions = binarize_scores(self.gtruth,
                                                  self.predictions,
                                                  self.num_class)

        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(self.num_class):
            precision[i], recall[i], _ = metrics.precision_recall_curve(gtruth[:, i],
                                                                predictions[:, i])
            average_precision[i] = metrics.average_precision_score(gtruth[:, i],
                                                           predictions[:, i])
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(gtruth.ravel(),
                                                                        predictions.ravel())
        average_precision["micro"] = metrics.average_precision_score(gtruth, predictions,
                                                             average="micro")

        if plot:
            self._plot_precision_recall_curve(precision, recall, average_precision)

        return precision, recall, average_precision

    def _plot_confusion_matrix(self, cm, cmap=plt.cm.Blues, save=False):
        """Plot the confusion matrix and diagonal class accuracies"""
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        fmt = '.2f'
        thresh = cm.max() / 2. if not math.isnan(cm.max()) else 50.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Plot diagonal scores alongside it
        plt.subplot(1, 2, 2)
        temp = dict(zip(self.classes, np.nan_to_num(cm.diagonal())))
        cm_dict = OrderedDict(sorted(temp.items(), key=lambda x:x[1]))
        classes = list(cm_dict.keys())
        cm_diag = list(cm_dict.values())

        ax = pd.Series(cm_diag).plot(kind='barh')
        ax.set_xlabel('Class Accuracy')
        ax.set_yticklabels(classes)
        rects = ax.patches
        # Make some labels.
        for rect, label in zip(rects, cm_diag):
            width = rect.get_width()
            label = np.nan if label == 0 else label
            ax.text(width + 5, rect.get_y() + rect.get_height() / 2, format(label, fmt),
                    ha='center', va='bottom')

        if save:
            cm_fname = os.path.join(opt.model_dir, 'figs', self.pre + '_confusion.png')
            plt.savefig(cm_fname)
        plt.show()

    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot the ROC Curve"""
        plt.figure(figsize=(20,15))
        lw = 2

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        if self.colors == None:
            self.colors = random.sample(plt_colors.cnames.keys(), k=self.num_class)
        for i, color in zip(range(self.num_class), self.colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Postive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_fig_name = os.path.join(opt.model_dir, 'figs', self.pre + '_roc_curve.png')
        plt.savefig(roc_fig_name)
        plt.show()

    def _plot_precision_recall_curve(self, recall, precision, average_precision):
        """Plot the Precision-Recall Curve"""
        plt.figure(figsize=(20,15))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))
        for i, color in zip(range(self.num_class), self.colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        plt.tight_layout()
        pr_fig_fname = os.path.join(opt.model_dir, 'figs', self.pre + '_precision-recall.png')
        plt.savefig(pr_fig_fname)
        plt.show()

    def save_predictions(self, start_datetime=None, run_time=0,
                         model_version=None, dest_dir=None):
        """Load predictions into json file"""
        from datetime import datetime
        date_fmt = '%Y-%m-%d_%H:%M:%S'

        # preprocess probabilities
        probabilities = [np.max(prob) for prob in self.probabilities]

        # construct prediction dataframe
        total_smpls = len(self.ids)
        pred_cols = [CONST.IMG, CONST.PRED, CONST.PROB, CONST.PRED_TSTAMP,
                     CONST.MODEL_NAME]
        data = [self.ids, self.predictions, probabilities,
                [datetime.today().strftime(date_fmt)]*total_smpls,
                [model_version]*total_smpls]
        pred_df = pd.DataFrame(dict(zip(pred_cols, data)))

        # decode predictions
        pred_df[CONST.PRED] = self.le.inverse_transform(pred_df[CONST.PRED].values)
        pred_df[CONST.HAB_PRED] = pred_df[CONST.PRED].apply(self.le.hab_map)

        # merge into original csv
        orig_df = pd.read_csv(opt.deploy_data)
        merged_df = orig_df.merge(pred_df, on=CONST.IMG)

        self.pre = os.path.basename(opt.deploy_data).strip('.csv')
        csv_fname = os.path.join(opt.model_dir, self.pre + '-predictions.csv')
        merged_df.to_csv(csv_fname, index=False)
        self.logger.info('Predictions saved to {}'.format(csv_fname))

    def print_eval(self, params, run_time, err, acc, results_dir, **kwargs):
        """Print evaluation metrics to log file

        Args:
            params (float): Model parameters
            run_time (float): Run time
            err (float): Loss
            acc (float): Accuracy
            results_dir (str): Abs path to the results directory
            **kwargs: Keyword arguments

        Returns:

        """
        logger = logging.getLogger('print_eval')
        log = '[PARAMETERS] {params}'.format(params=params)
        logger.info(log)
        log = '[RUN TIME] {time.avg:.3f} sec/sample'.format(time=run_time)
        logger.info(log)
        log = '[FINAL] {name:<30} {loss:.7f}'.format(
            name='{}/{}'.format(opt.mode.upper(), 'crossentropy'), loss=err)
        logger.info(log)
        log = '[FINAL] {name:<30} {acc:.7f}'.format(
            name='{}/{}'.format(opt.mode.upper(), 'accuracy'), acc=acc)
        logger.info(log)

        # Log any other given keyword arguments
        for k, v in kwargs.items():
            log = '[FINAL] {name:<30} {val}'.format(
                name='{}/{}'.format(opt.mode.upper(), k), val=v)
            logger.info(log)
        log = '[FIGS] {}'.format(results_dir)

        # Compute classification report
        logger.info(metrics.classification_report(self.gtruth, self.predictions,
                                          target_names=self.le.hab_classes))
        logger.info(log)

    def _plot_dataset_distribution(self, data_dict):
        """Plot dataset distribution in the form of raw counts and normalized counts"""
        # Grab data from dictionary
        labels = list(data_dict.keys())
        sizes = list(data_dict.values())
        num_samples = range(len(labels))
        if self.colors == None:
            self.colors = random.sample(plt_colors.cnames.keys(), k=len(labels))

        # Generate plot
        plt.figure(figsize=(20,8))
        plt.subplot(1,2,1)
        bars = plt.bar(num_samples, sizes)
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(num_samples, labels, rotation=20, fontsize=12)

        # Assign text to top of bars
        for rect in bars:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % int(height),
                     ha='center', va='bottom', fontsize=14)

        # Plot Pie chart of the distribution
        plt.subplot(1,2,2)
        plt.pie(sizes, labels=labels, colors=self.colors,
                autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize':
                                                                               14})
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def vis_training(train_points, val_points, num_epochs=0, loss=True, **kwargs):
    """ Visualize losses and accuracies w.r.t each epoch

    Args:
        num_epochs: (int) Number of epochs
        train_points: (list) Points of the training curve
        val_points: (list) Points of the validation curve
        loss: (bool) Flag for loss or accuracy. Defaulted to True for loss

    """
    # Check if nan values in data points
    train_points = [i for i in train_points if not math.isnan(i)]
    val_points = [i for i in val_points if not math.isnan(i)]
    num_epochs = len(train_points)
    x = np.arange(0, num_epochs)

    plt.figure()
    plt.plot(x, train_points, 'b')
    plt.plot(x, val_points, 'r')

    title = '{} vs Number of Epochs'.format('Loss' if loss else 'Accuracy')
    if 'EXP' in kwargs:
        title += ' (EXP: {})'.format(kwargs['EXP'])
    plt.title(title)

    if loss:
        plt.ylabel('Cross Entropy Loss')
    else:
        plt.ylabel('Accuracy')

    plt.gca().legend(('Train', 'Val'))
    plt.xlabel('Number of Epochs')

    figs_folder_path = os.path.join(opt.model_dir, 'figs')
    if not os.path.exists(figs_folder_path):
        os.makedirs(figs_folder_path)
    
    save_path = os.path.join(opt.model_dir,'figs/train_val_{}'.format('loss' if loss else 'accuracy'))
    #save_path = './figs/train_val_{}'.format('loss' if loss else 'accuracy')
    for k_, v_ in kwargs.items():
        save_path += '_%s' % v_
    save_path += '.png'

    plt.savefig(save_path)
    plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def binarize_scores(gtruth, predictions, num_class):
    """One hotencode gtruth and predictions"""
    from sklearn import preprocessing
    gtruth = np.asarray(gtruth).reshape(-1, 1)
    predictions = np.asarray(predictions).reshape(-1, 1)

    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.array(range(num_class)).reshape(-1, 1))

    gtruth = enc.transform(gtruth).toarray()
    predictions = enc.transform(predictions).toarray()
    return gtruth, predictions
