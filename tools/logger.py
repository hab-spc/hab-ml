import sys
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np

# Constants
ROOT = os.path.abspath(os.path.dirname(__file__))

class Logger(object):
    def __init__(self,
                 name='default',
                 debug=False,
                 version=None,
                 saveDir=None,
                 description=None,
                 autosave=False):

        if saveDir is not None:
            global ROOT
            ROOT = saveDir

        self.name = name
        self.metrics = []
        self.tags = {}
        self.version = version
        self.description = description
        self.debug = debug
        self.autosave = autosave
        self.createdAt = str(datetime.utcnow())

        if version is None:
            oldVersion = self.getLastExpVersion()
            self.version = oldVersion + 1

        self.initFile()

        if not self.debug:

            if self.version is not None:

                if not os.path.exists(self.getLogName()):
                    self.createExpFile(self.version)
                    self.save()
                else:
                    self.load()
            else:
                oldVersion = self.getLastExpVersion()
                self.version = oldVersion
                self.createExpFile(self.version+1)
                self.save()
        self.saveDir = self.getDataPath(self.name, self.version)

    def initFile(self):
        expFile = self.getDataPath(self.name, self.version)
        if not os.path.isdir(expFile):
            os.makedirs(expFile)

    def tag(self, tagDict):
        if self.debug: return

        if 'createdAt' not in tagDict:
            self.tags['createdAt'] = str(datetime.utcnow())

        for key, val in tagDict.items():
            self.tags[key] = val
        # self.tags.append(tagDict)

        if self.autosave == True:
            self.save()

    def log(self, metricsDict):
        if self.debug: return

        if 'createdAt' not in metricsDict:
            metricsDict['createdAt'] = str(datetime.utcnow())

        self.metrics.append(metricsDict)

        if self.autosave == True:
            self.save()

    def save(self):
        metricsFilePath = self.getDataPath(self.name, self.version) + '/metrics.csv'
        metaFilePath = self.getDataPath(self.name, self.version) + '/meta.txt'

        obj = {
            'name': self.name,
            'version': self.version,
            'metaPath': metricsFilePath,
            'metricsPath': metaFilePath,
            'description': self.description,
            'createdAt': self.createdAt,
            'autosave': self.autosave
        }

        with open(self.getLogName(), 'w') as f:
            json.dump(obj, f, ensure_ascii=False)

        # metadf = pd.DataFrame({'key': list(self.tags.keys()), 'value': list(self.tags.values())})
        # # metadf = pd.DataFrame(self.tags)
        # metadf.to_csv(metaFilePath, index=False)
        with open(metaFilePath, 'w') as metaFile:
            for key, val in self.tags.items():
                metaFile.write('{}: {}\n'.format(key,val))
            metaFile.write('\n\n')
        f.close()

        metricsdf = pd.DataFrame(self.metrics)
        metricsdf.to_csv(metricsFilePath, index=False)

    def load(self):
        with open(self.getLogName(), 'r') as f:
            data = json.load(f)
            self.name = data['name']
            self.version = data['version']
            self.autosave = data['autosave']
            self.createdAt = data['createdAt']
            self.description = data['description']

        metaFilePath = self.getDataPath(self.name, self.version) + '/meta.txt'
        for line in open(metaFilePath, 'r').read().splitlines():
            if line:
                key, value = line.split (': ')
                try:
                    value = int (value)
                except ValueError:
                    try:
                        value = float (value)
                    except ValueError:
                        pass
                self.tags[key] = value
        # df = pd.read_csv(metaFilePath)
        # self.tags = df.to_dict(orient='records')

        metricsFilePath = self.getDataPath(self.name, self.version) + '/metrics.csv'

        df = pd.read_csv(metricsFilePath)
        self.metrics = df.to_dict(orient='records')

        # remove nans
        for metric in self.metrics:
            to_delete = []
            for k, v in metric.items ():
                try:
                    if np.isnan (v):
                        to_delete.append (k)
                except Exception as e:
                    pass

            for k in to_delete:
                del metric[k]


    def getDataPath(self, expName, expVersion):
        return os.path.join(ROOT, expName, 'version_{}'.format(expVersion))

    def getLogName(self):
        expFile = self.getDataPath(self.name, self.version)
        return '{}/meta.experiment'.format(expFile)

    def createExpFile(self, version):
        expFile = self.getDataPath(self.name, self.version)
        open('{}/meta.experiment'.format(expFile), 'w').close()
        self.version = version

    def getLastExpVersion(self):
        try:
            expFile = os.sep.join(self.getDataPath(self.name, self.version).split(os.sep)[:-1])
            last_version = -1
            for f in os.listdir (expFile):
                if 'version_' in f:
                    file_parts = f.split ('_')
                    version = int (file_parts[-1])
                    last_version = max (last_version, version)
            return last_version
        except Exception as e:
            return -1



if __name__ == '__main__':
    exp = Logger(name='randomForest',
                 saveDir='/Users/ktl014/PycharmProjects/PersonalProjects/EmployeeTurnOverPrediction/git/records',
                 version=1,
                 autosave=False)
    exp.tag ({'dataset_name': 'imagenet_1', 'learning_rate': 0.0001, 'classes':1})
    exp.log ({'val_loss': 0.22, 'epoch_nb': 1, 'batch_nb': 12})
    exp.log ({'tng_loss': 0.01})
    exp.save ()
