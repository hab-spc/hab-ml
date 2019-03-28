'''
utils

Created on May 14 2018 23:15 
#@author: Kevin Le 
'''
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from collections import OrderedDict
import os

from dataset import SPCDataset

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

def save_predictions(img_paths, gtruth, predictions, probs, label_file, output_fn):
    # df = pd.DataFrame({
    #     'image':img_paths,
    #     'gtruth':gtruth,
    #     'predictions':predictions,
    #     'confidence_level': calculate_confidence_lvl(probs)
    # })
    data = OrderedDict()
    data['image'] = img_paths
    data['gtruth'] = gtruth
    data['predictions'] = predictions
    data['confidence_level'] = calculate_confidence_lvl(probs)
    df = pd.DataFrame(data)
    df = map_labels(df, label_file, 'predictions')
    try:
        df.to_csv(output_fn)
    except:
        print('{} does not exist'.format(output_fn))

def calculate_confidence_lvl(probs):
    side_lobe = np.sort(probs)
    side_lobe = side_lobe[::-1]
    confidence_level = [(side_lobe[i,1]-side_lobe[i,0])/side_lobe[i,1] for i in range(len(side_lobe))]
    return confidence_level

def verify_dates():
    import glob
    import os

    data_dict = {}
    for data in ['good_proro', 'bad_proro']:
        img = {}
        images = glob.glob (
            '/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton/images_orig/{}/*'.format(data))
        img['images'] = sorted([os.path.basename (i).replace ('jpg', 'tif') for i in images])
        img['times'] = sorted([i.split('-')[1] for i in img['images']])
        data_dict[data] = img
    data_dict['bad_proro']['spc2'] = [i.split('-')[1] for i in data_dict['bad_proro']['images'] if 'SPC2' in i]
    data_dict['bad_proro']['spcp2'] = [i.split ('-')[1] for i in data_dict['bad_proro']['images'] if 'SPCP2' in i]

def main():
    pass

if __name__ == '__main__':
    main()