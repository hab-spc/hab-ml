'''
create_lmdb

Created on May 09 2018 11:47 
#@author: Kevin Le 
'''
from __future__ import print_function, division

import cv2
import lmdb
import caffe
import numpy as np
import os
import argparse
import sys
import time

from preprocessing import aspect_resize, convert_to_8bit
from dataset import SPCDataset

DEBUG = True
ALL = ['train', 'val', 'test']

def parse_cmd():
    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('-d', '--dataset', default=1, help='Dataset. Version: 1')
    parser.add_argument('--phase', nargs='+', type=str, default=ALL, help='Type of dataset. Allows one or more types to be created. Options: train, val, test')
    parser.add_argument('--root', default='/data6/lekevin/phytoplankton')
    parser.add_argument('--image_dir', default='/data6/lekevin/phytoplankton/rawdata', help=' Directory containing all images. Images should be stored at '
                        ' {BASEDIR}/{DAY}/{FN} where file names FN are given by the lines of the csv file.')
    parser.add_argument('--output', '-o', default=None, help='Output LMDB directory name. Raises exception if directory already exists')

    args = parser.parse_args(sys.argv[1:])
    return args

def write_caffe_lmdb(img_fns, lbls, output_fn):
    def preprocessing(img):
        img = (img*255).astype(np.uint8)
        # img = convert_to_8bit(img)
        img = aspect_resize(img)
        img = img[:,:, (2,1,0)]
        img = np.transpose(img, (2,0,1))
        return img

    if os.path.exists (output_fn):
        raise ValueError (output_fn + ' already exists!')

    nSmpls = len(img_fns)
    map_size = nSmpls*3*256*256*8*1.5
    env_img = lmdb.open(output_fn, map_size=map_size)
    for i in range(nSmpls):
        try:
            img = preprocessing(caffe.io.load_image(img_fns[i]))

            # Write image datum
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels, datum.height, datum.width = img.shape[0], img.shape[1], img.shape[2]
            datum.data = img.tostring()
            datum.label = int(lbls[i])

            with env_img.begin(write=True) as txn:
                txn.put('{:08}'.format(i).encode('ascii'), datum.SerializeToString())

            print('Samples saved {}/{} ({:.0f}%)     \r'.format(i, nSmpls, 100.0*i/nSmpls), end='')
        except:
            print(img_fns[i])


def main(args):
    since = time.time()

    # Try to find exisiting data file path and create LMDB under the version
    csv_filename = os.path.join(args.root, 'data', str(args.dataset), 'data_{}.csv')
    for phase in args.phase:
        dataset = SPCDataset(csv_filename=csv_filename.format(phase), img_dir=args.image_dir, phase=phase)
        print(dataset)

        fns, lbls = dataset.get_fns()

        if args.output is None:
            output = os.path.join(args.root, 'data', str(args.dataset))
        else:
            output = args.output

        if not os.path.exists(output):
            os.makedirs(output)

        lmdb_fn = os.path.join(output, '{}.LMDB'.format(phase))
        write_caffe_lmdb(img_fns=fns, lbls=lbls, output_fn=lmdb_fn)
        print('{} written Time elapsed {:.0f}s'.format(os.path.basename(lmdb_fn), time.time() - since))

    time_elapsed = time.time() - since
    print('Dataset generated in {:.0f}h, {:.0f}m'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60))

if __name__ == '__main__':
    main(parse_cmd())