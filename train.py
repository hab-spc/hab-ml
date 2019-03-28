'''
train

Created on May 11 2018 11:25 
#@author: Kevin Le 
'''
import argparse
import caffe
import datetime
import os
import sys
import time

from tools.dataset import SPCDataset
from tools.logger import Logger
from model import ClassModel

DEBUG=False

def parse_cmds():
    parser = argparse.ArgumentParser(description='Train Classification Model')
    parser.add_argument('--root', default='/data6/lekevin/phytoplankton/')
    parser.add_argument('--dataset', type=str, default=1, help='Dataset version')
    parser.add_argument('--model_name', type=str, default='caffenet')
    parser.add_argument('--img_dir', default='/data6/lekevin/phytoplankton/rawdata', help='Image directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--exp', '-e', default=1, help='Experiment version')
    parser.add_argument('-d', '--description', default=None, help='Description of trained model')

    parser.add_argument('--test_iters', type=int, default=10000, help='Test iterations')
    args = parser.parse_args(sys.argv[1:])
    return args

def train_model(args, LMDBs):
    model_weights = {'caffenet':'caffenet/bvlc_reference_caffenet.caffemodel',
                     'resnet-50':'resnet-50/ResNet-50-model.caffemodel',
                     'vgg19':'vgg19/VGG_ILSVRC_19_layers.caffemodel'}
    solver_proto = args.root + 'caffe/{}/solver.prototxt'.format(args.model_name)
    train_proto = args.root + 'caffe/{}/train_val.prototxt'.format(args.model_name)
    weights = args.root + 'caffe/{}'.format(model_weights[args.model_name])
    model_filename = os.path.join(args.root, 'records', args.model_name, 'version_{}'.format(args.exp), 'model.caffemodel')
    if DEBUG:
        print(solver_proto, weights, model_filename)
        exit(0)

    model = ClassModel()
    model.prep_for_training(solver_proto=solver_proto, train_proto=train_proto, weights=weights, LMDBs=LMDBs, gpu_id=args.gpu)

    test_iters = args.test_iters
    since = time.time()
    try:
        for i in range(test_iters):
            model.train(n=1)
    except KeyboardInterrupt:
        print('Training interrupted. Current model will be saved at {}'.format(model_filename))
    finally:
        model.save(model_filename)
        time_elapsed = time.time() - since
        print("Training completed in {:.0f}h {:.0f}m".format(time_elapsed//3600, (time_elapsed%3600)//60))

def main(args):
    print('Initiailizing Logger...')
    exp = Logger(name=args.model_name,
                 version=args.exp,
                 saveDir=os.path.join(args.root, 'records'),
                 description=args.description,
                 autosave=True)
    exp.log({'createdAt': time.time(), 'phase':'train'})
    #TODO write savedir to meta file and have read option

    # Select LMDBs
    csv_filename = os.path.join(str(args.root), 'data', str(args.dataset), 'data_{}.csv')
    dataset = {phase: SPCDataset(csv_filename=csv_filename.format(phase), img_dir=args.img_dir, phase=phase) for phase in ['train', 'val']}
    LMDBs = {phase: dataset[phase].get_lmdbs() for phase in ['train','val']}

    # train model
    train_model(args, LMDBs)


if __name__ == '__main__':
    main(parse_cmds())