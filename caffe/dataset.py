'''
dataset

Created on Jun 04 2018 11:06 
#@author: Kevin Le 
'''
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
import numpy as np

ROOT = '/data6/lekevin/phytoplankton'

def create_dataset(version):
    image_dir = ROOT + '/rawdata/images/'
    df = pd.read_csv(ROOT + '/rawdata/data.csv')
    df = df[['img', 'img_label', 'day']].copy()
    df = df.rename(columns={'img': 'image', 'img_label':'label'})
    df['image'] = image_dir + df['image']
    df['day'] = df['day'].map({'Mon Mar 20': 'day1', 'Mon Mar 27': 'day2', 'Mon Apr 10': 'day3'})
    df['label'] = df['label'].apply(lambda x: 1 if 'Prorocentrum' in x else 0)

    test = df[df['day'] == 'day3']
    drop_nimgs = test['label'].value_counts()[0] - test['label'].value_counts()[1]
    test = test.drop(test[test['label'] == 0].sample(n=drop_nimgs).index)

    df = df.drop(test.index)
    nimg_diff = df['label'].value_counts()[0] - df['label'].value_counts()[1]    # difference in number of images to drop
    df = df.drop(df[df['label'] == 0].sample(n=nimg_diff).index)

    train, val, _, _ = train_test_split(df, df['label'], test_size=0.15, random_state=42)

    dest_path = os.path.join (ROOT, 'data/{}'.format (version))
    if not os.path.exists (dest_path):
        os.makedirs (dest_path)

    with open(dest_path + '/stats.txt', 'w') as f:
        dataset = {'train': train, 'val': val, 'test': test}
        for phase in ['train', 'val', 'test']:
            dataset[phase].to_csv(os.path.join(dest_path, 'data_{}.csv'.format(phase)))
            print('Dataset written to {}'.format(os.path.join(dest_path, 'data_{}.csv'.format(phase))))
            f.write('{} dataset\n'.format(phase))
            for key, val in dataset[phase]['label'].value_counts().to_dict().iteritems():
                f.write('{}: {}\n'.format(key, val))
    f.close()

class SPCDataset(object):

    def __init__(self, csv_filename, img_dir, phase):
        self.data = pd.read_csv(csv_filename)
        self.img_dir = img_dir
        self.data_dir = os.path.dirname(csv_filename)
        self.phase = phase
        self.size = self.data.shape[0]
        self.numclasses = len(self.data['label'].unique())
        self.lmdb_path = os.path.join(self.data_dir, '{}.LMDB'.format(self.phase))

        #TODO Give option to user to create lmdb after making dataset if he wants to
        # if not os.path.exists(self.lmdb_path):


    def __repr__(self):
        return 'Dataset [{}] {} classes, {} images\n{}'.\
            format(self.phase, self.numclasses, self.size, self.data['label'].value_counts())

    def get_fns(self):
        '''
        Return filenames and labels
        :return: fns: list, lbls: array
        '''
        shuffle_images = (self.phase == 'train' or self.phase == 'val')
        if shuffle_images:
            self.data = self.data.iloc[np.random.permutation(self.size)]
            self.data = self.data.reset_index(drop=True)

        self.fns = list(self.data['image'])
        self.lbls = np.array(self.data['label'])
        return self.fns, self.lbls

    def get_lmdbs(self):
        '''
        Get LMDBs (dataset version)
        :return: LMDB path
        '''
        # Catch if LMDBs exist or not
        try:
            return self.lmdb_path
        except:
            print('{} was not found or does not exist!'.format(self.lmdb_path))

    def load_lmdb(self):
        import lmdb
        import caffe

        '''
        Load LMDB
        :param fn: filename of lmdb
        :return: images and labels
        '''
        print ("Loading " + str (self.lmdb_path))
        env = lmdb.open (self.lmdb_path, readonly=True)
        datum = caffe.proto.caffe_pb2.Datum ()
        with env.begin () as txn:
            cursor = txn.cursor ()
            data, labels = [], []
            for _, value in cursor:
                datum.ParseFromString (value)
                labels.append (datum.label)
                data.append (caffe.io.datum_to_array (datum).squeeze ())
        env.close ()
        print ("LMDB successfully loaded")
        return data, labels


if __name__ == '__main__':
    version = 1
    create_dataset(version=version)
    # root = '/data6/lekevin/phytoplankton'
    # img_dir = '/data6/lekevin/phytoplankton/rawdata'
    # csv_filename = os.path.join (root, 'data', str (version), 'data_{}.csv')
    #
    # # Test initialization
    # dataset = {phase: SPCDataset (csv_filename=csv_filename.format (phase), img_dir=img_dir, phase=phase) for phase in
    #            ['train', 'val', 'test']}
    # for phase in dataset:
    #     print (dataset[phase])