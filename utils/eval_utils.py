import itertools
import json
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import logging
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
        self.softmax = nn.Softmax()

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
            confusion_matrix_rate[i, :] = (confusion_matrix_rawcount[i, :]) / \
                                          class_count[i, 0] * 100
        confusion_matrix_rate = np.around(confusion_matrix_rate, decimals=4)
        if plot:
            self._plot_confusion_matrix(confusion_matrix_rate)
        return confusion_matrix_rate


    def _plot_confusion_matrix(self, cm, normalize=False,
                               title='Confusion matrix',  cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

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

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.results_dir + '/cm.png')
        plt.show()

    def save_predictions(self, start_datetime=None, run_time=0,
                         model_version=None, dest_dir=None):
        """Load predictions into json file"""
        from datetime import datetime
        import json
        date_fmt = '%Y-%m-%d_%H:%M:%S'

        key_fmt = ['image_id', 'gtruth', 'pred', 'prob']
        total_smpls = len(self.ids)
        json_dict = {
            'start-datetime': start_datetime,
            'end-datetime': datetime.today().strftime(date_fmt),
            'total-samples': total_smpls,
            'run_time': run_time,
            'model_version': model_version,
            'machine_labels': [],
        }
        for i in range(total_smpls):
            dd = [self.ids[i], int(self.gtruth[i]), int(self.predictions[i]),
                  self.probabilities[i]]
            key_fmt = ['image_id', 'gtruth', 'pred', 'prob']
        # dd = [self.ids[i], int(self.gtruth[i]), int(self.predictions[i])]
            params = dict(zip(key_fmt, dd))
            json_dict['machine_labels'].append(params)

        # Save as json file
        json_fname = os.path.join(dest_dir, 'predictions.json')
        with open(json_fname, 'w', encoding='utf-8') as json_file:
            json.dump(json_dict, json_file, indent=4, separators=(',', ':'),
                      sort_keys=True, cls=NumpyEncoder)
        json_file.close()
        print('Predictions saved to {}'.format(json_fname))


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

    if not os.path.exists('figs'):
        os.makedirs('figs')

    save_path = './figs/train_val_{}'.format('loss' if loss else 'accuracy')
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