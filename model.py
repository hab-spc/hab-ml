'''
model

Created on Jun 04 2018 17:49 
#@author: Kevin Le 
'''
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import numpy as np

class ClassModel(object):
    def __init__(self):
        pass

    def prep_for_training(self, solver_proto, train_proto, weights, LMDBs, gpu_id):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        #self.append_proto(proto=train_proto, LMDB=LMDBs)

        self.solver = caffe.SGDSolver(solver_proto)
        self.solver.net.copy_from(weights)

    def load_solver_proto(self):
        pass

    def train(self, n=1):
        self.solver.step(n)

    def load(self):
        pass

    def save(self, model_fn):
        self.solver.net.save(model_fn)

    def append_proto(self, proto, LMDB=None):
        assert 'train' and 'val' in LMDB

        net = caffe_pb2.NetParameter()
        new_proto = caffe_pb2.NetParameter()
        with open(proto, 'r') as f:
            s = f.read()
        txtf.Merge(s, net)

        for i in range (0, len (net.layer)):
            new_proto.layer.extend ([net.layer[i]])
            if net.layer[i].name == 'data' and 'train' in new_proto.layer[i].data_param.source:
                new_proto.layer[i].data_param.source = LMDB['train']
            elif net.layer[i].name == 'data' and 'val' in new_proto.layer[i].data_param.source:
                new_proto.layer[i].data_param.source = LMDB['val']

        new_proto = '/data6/lekevin/cayman/caffe/train_val.prototxt'
        with open (new_proto, 'w') as f:
            f.write (txtf.MessageToString (new_proto))

    def prep_for_deploy(self, deploy_proto, weights, gpu_id):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.deploy = caffe.Net(deploy_proto, caffe.TEST, weights=weights)

    def forward(self, batch, batch_size):
        self.deploy.blobs['data'].data[:batch_size] = batch
        self.deploy.forward()
        return np.copy(self.deploy.blobs['prob'].data[:batch_size,:])

def main():
    pass


if __name__ == '__main__':
    pass
