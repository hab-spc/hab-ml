import itertools
import json
import logging
import math
import os
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

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

def print_eval(params, run_time, err, acc, results_dir, **kwargs):
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

    for k,v in kwargs.items():
        log = '[FINAL] {name:<30} {val}'.format(
            name='{}/{}'.format(opt.mode.upper(), k), val=v)
        logger.info(log)
    log = '[FIGS] {}'.format(results_dir)
    logger.info(log)

def get_meter(meters=['batch_time', 'loss', 'acc']):
    return {meter_type: AverageMeter() for meter_type in meters}

def accuracy(predictions, targets, axis=1):
    batch_size = predictions.size(0)
    predictions = predictions.max(axis)[1].type_as(targets)
    hits = predictions.eq(targets)
    acc = 100. * hits.sum().float() / float(batch_size)
    return acc

class EvalMetrics(object):
    def __init__(self, classes, predictions=[], gtruth=[], ids=[],
                 model_dir=''):
        self.num_class = len(classes)
        self.classes = classes
        self.predictions = predictions
        self.gtruth = gtruth
        self.ids = ids
        self.results_dir = os.path.join(model_dir, 'figs')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.probabilities = []
        self.softmax = nn.Softmax(dim=1)

        self.le = HABLblEncoder()

        self.logger = logging.getLogger(__name__)

    def update(self, predictions, targets, ids, axis=1):
        # TODO save image id along with predicitons to save back to image files
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
        self.num_class = 10
        self.classes = self.le.hab_classes
        self.gtruth = self.le.hab_transform2idx(self.gtruth)
        self.predictions = self.le.hab_transform2idx(self.predictions)

        c=0
        for t, p in zip(self.gtruth, self.predictions):
            if t==p:
                c+=1
        return c/len(self.gtruth)*100

    def compute_cm(self, plot=False):
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
            if class_count[i, 0] == 0:
                confusion_matrix_rate[i, :] = 0
            else:
                confusion_matrix_rate[i, :] = (confusion_matrix_rawcount[i, :]) / \
                                                  class_count[i, 0] * 100
            
        confusion_matrix_rate = np.around(confusion_matrix_rate, decimals=4)
        if plot:
            self._plot_confusion_matrix(confusion_matrix_rate)
        return confusion_matrix_rate


    def _plot_confusion_matrix(self, cm, normalize=False,
                                cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized confusion matrix"
        else:
            title = 'Confusion matrix, without normalization'

        plt.figure(figsize=(30,30))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        cm_fname = os.path.join(opt.model_dir, 'figs', os.path.basename(
                opt.deploy_data).strip('.csv') + '_confusion.png')
        plt.savefig(cm_fname)
        plt.show()
        
        #Store diagonal of the confusion matrix
        cm_diag = []
        for i in range(cm.shape[0]):
            cm_diag.append(cm[i,i])
        class_diag = self.classes.copy()
        class_diag = [x for _,x in sorted(zip(cm_diag,class_diag))]
        cm_diag.sort()
        cm_diag_fname = cm_fname.replace('_confusion.png', '_confusion_diag.txt')
        with open(cm_diag_fname, 'w') as the_file:
            for i in range(len(cm_diag)):
                line = '{:>39}  {:>12}'.format(class_diag[i], str(cm_diag[i]))
                the_file.write(line+'\n')
        

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

        csv_fname = os.path.join(
            opt.model_dir, os.path.basename(opt.deploy_data).strip('.csv') +
                           '-predictions.csv')
        merged_df.to_csv(csv_fname, index=False)
        self.logger.info('Predictions saved to {}'.format(csv_fname))


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

def plot_roc_curve(gtruth, predictions, num_class):
    from sklearn import metrics
    from sklearn import preprocessing
    from scipy import interp

    enc = preprocessing.OneHotEncoder ()
    enc.fit (gtruth.reshape (-1, 1))
    gtruth = enc.transform (gtruth.reshape (-1, 1)).toarray ()

    enc1 = preprocessing.OneHotEncoder ()
    enc1.fit (predictions.reshape (-1, 1))
    predictions = enc1.transform (predictions.values.reshape (-1, 1)).toarray ()

    fpr, tpr = {}, {}
    roc_auc = {}
    for i in range (num_class):
        fpr[i], tpr[i], _ = metrics.roc_curve (gtruth[:, i], predictions[:, i])
        roc_auc[i] = metrics.auc (fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve (gtruth.ravel (), predictions.ravel ())
    roc_auc["micro"] = metrics.auc (fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique (np.concatenate ([fpr[i] for i in range (num_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like (all_fpr)
    for i in range (num_class):
        mean_tpr += interp (all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc (fpr["macro"], tpr["macro"])

    plt.figure ()
    lw = 2

    # Plot all ROC curves
    plt.figure ()
    plt.plot (fpr["micro"], tpr["micro"],
              label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format (roc_auc["micro"]),
              color='deeppink', linestyle=':', linewidth=4)

    plt.plot (fpr["macro"], tpr["macro"],
              label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format (roc_auc["macro"]),
              color='navy', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange']
    for i, color in zip (range (num_class), colors):
        plt.plot (fpr[i], tpr[i], color=color, lw=lw,
                  label='ROC curve of class {0} (area = {1:0.2f})'.format (i, roc_auc[i]))
    plt.plot ([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim ([0.0, 1.0])
    plt.ylim ([0.0, 1.05])
    plt.xlabel ('False Postive Rate')
    plt.ylabel ('True Positive Rate')
    plt.title ('Receiver Operating Characteristic')
    plt.legend (loc="lower right")
    plt.show ()