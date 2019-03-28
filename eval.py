'''
eval

Created on May 14 2018 17:32 
#@author: Kevin Le 
'''
from __future__ import print_function, division

import argparse
import sys
import os
import time
import numpy as np
import cPickle as pickle

from model import ClassModel
from tools.dataset import SPCDataset
from tools.logger import Logger
from tools.utils import save_predictions


def parse_cmds():
    parser = argparse.ArgumentParser(description='Train Cayman Classification Model')
    parser.add_argument('--root', default='/data6/lekevin/cayman/')
    parser.add_argument('--phase', default='val', type=str, help='')
    parser.add_argument('--dataset', type=str, default=1, help='Dataset version')
    parser.add_argument('--model_name', type=str, default='model_d1')
    parser.add_argument('--img_dir', default='/data6/lekevin/cayman/rawdata', help='Image directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--exp', '-e', default=1, help='Experiment version')
    parser.add_argument('-d', '--description', default=None, help='Description of trained model')

    args = parser.parse_args(sys.argv[1:])
    return args

def evaluate_model(args, db_object, log_object):
    from sklearn.metrics import roc_auc_score
    try:

        deploy_proto = args.root + 'caffe/{}/deploy.prototxt'.format(args.model_name)
        model_dir = os.path.join(args.root, 'records', args.model_name, 'version_{}'.format(args.exp))
        selected_weights = [os.path.join(model_dir,i) for i in os.listdir(model_dir) if i.endswith('.caffemodel')][-1]
    except:
        print(deploy_proto, model_dir, selected_weights)
    model = ClassModel()
    model.prep_for_deploy(deploy_proto=deploy_proto, weights=selected_weights, gpu_id=args.gpu)

    images, labels = db_object.load_lmdb()

    probs = []
    nSmpl = len(images)

    # Set up input preprocessing
    for i in range(0,nSmpl,25):

        def prep_image(img):
            img = img.astype (float)[:, 14:241, 14:241]  # center crop (img is in shape [C,X,Y])
            img -= np.array ([104., 117., 123.]).reshape ((3, 1, 1))  # demean (same as in trainval.prototxt
            return img

        # Configure preprocessing
        batch = [prep_image(img) for img in images[i:i +25]]
        batch_size = len(batch)

        output = model.forward(batch, batch_size)
        probs.append(output)

        print('Samples computed {} / {} ({:.0f}%)     \r'.format(i, nSmpl, 100.0*i/nSmpl), end='')
    print()

    #TODO figure out how to output top 5 predictions
    probs = np.concatenate(probs, 0)
    predictions = probs.argmax(1)
    gtruth = np.array(labels)
    accuracy = (gtruth == predictions).mean()*100
    # auc = roc_auc_score(gtruth, predictions)
    eval_metrics = {'acc': accuracy, 'phase':args.phase} #, 'auc': auc}

    log_object.log(eval_metrics)
    log_object.save()
    img_paths, _ = db_object.get_fns()
    label_file = os.path.join(str(args.root), 'data', str(args.dataset), 'labels.txt')
    save_predictions(img_paths, gtruth, predictions, probs, label_file, log_object.saveDir + '/{}_predictions.csv'.format(args.phase))

    return eval_metrics

    #TODO tools
    '''
    2) output CM
    3) output ROC Curve
    4) output predictions to json
    '''


def print_eval(eval_metrics):
    print('Evaluation Results')
    for metric in eval_metrics:
        if not isinstance(eval_metrics[metric], (int,float)):
            continue
        print('{}: {:0.3f}'.format(metric, eval_metrics[metric]))

def main(args):
    since = time.time()
    print ('Initiailizing Logger...')
    exp = Logger (name=args.model_name,
                  version=args.exp,
                  saveDir=os.path.join (args.root, 'records'),
                  description=args.description,
                  autosave=True)

    # Select LMDBs
    csv_filename = os.path.join(str(args.root), 'data', str(args.dataset), 'data_{}.csv')
    dataset = {phase: SPCDataset(csv_filename=csv_filename.format(phase), img_dir=args.img_dir, phase=phase) for phase in [args.phase]}

    # Evaluate model
    eval_metrics = evaluate_model(args, db_object=dataset[args.phase], log_object=exp)
    print_eval(eval_metrics)
    elapsed_time = time.time() - since
    print('Evaluation completed in {:.0f}h {:.0f}m'.format(elapsed_time//3600, (elapsed_time%3600)//60))

if __name__ == '__main__':
    main(parse_cmds())