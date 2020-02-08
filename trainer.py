"""Trainer"""
# Standard dist imports
import logging
import os
import random

# Third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.resnet import freezing_layers
# Project level imports
from utils.config import opt
from utils.constants import Constants as CONST
from utils.eval_utils import get_meter
from utils.model_sql import model_sql


# Module level constants

class Trainer(object):
    """Trains Model

    Trainer is essentially a wrapper around the model, optimizer,
    and criterion. It assists with choosing which optimizer to use through
    flags, typecasting variables to cuda, and loading/saving checkpoints.

    """

    def __init__(self, model, model_dir=opt.model_dir, mode=CONST.TRAIN,
                 resume=opt.resume, lr=opt.lr, momentum=opt.momentum, class_count=None):
        """ Initialize Trainer

        Args:
            model: (MusicRNN) Model
            model_dir: (str) Path to the saved model directory
            mode: (str) Train or Test
            resume: (str) Path to pretrained model

        """
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(opt.logging_level)
        self.computing_device = self._set_cuda()

        self.model = model.to(self.computing_device)
        self.logger.debug("Model on CUDA? {}".format(next(self.model.parameters()).is_cuda))
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.start_epoch = 0
        self.best_err = np.inf

        self.optimizer = self._get_optimizer(lr=lr, momentum=momentum)
        
        # Defaulted to CrossEntropyLoss
        #TODO set interactive mode for setting the losses
        if opt.mode == CONST.TRAIN:
            self.logger.debug(class_count)
            if opt.interactive:
                weighted_y_n = input('Do you want to use weighted loss? (y/n)\n')
            else:
                weighted_y_n = opt.weighted_loss

            if weighted_y_n or weighted_y_n == 'y':
                weight = np.array([x for _,x in sorted(zip(class_count.keys().tolist(),class_count.tolist()))])
                self.logger.info('Class_count is: '+ str(weight))
                weight = weight/sum(weight)
                self.logger.info('Classes Weights are: '+ str(weight))
                weight = np.flip(weight).tolist()
                self.logger.info('Weighted Loss will be: ' + str(weight))
                weight = torch.FloatTensor(weight).cuda()
                self.criterion = nn.CrossEntropyLoss(weight=weight)
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.coral_criterion = CORAL

        # meter
        self.meter = {CONST.TRAIN: get_meter(), CONST.VAL: get_meter()}

        if resume or mode == CONST.VAL:
            sql = model_sql()
            fn = os.path.join(opt.model_dir, 'model_best.pth.tar')
            sql.close()
            self.load_checkpoint(fn)

        if mode == CONST.TRAIN:
            freezing_layers(model)

        if mode == CONST.DEPLOY:
            fn = os.path.join(opt.model_dir, 'model_best.pth.tar')
            self.load_checkpoint(fn)


    def generate_state_dict(self, epoch, best_err):
        """Generate state dictionary for saving weights"""
        return {
                'epoch': epoch+1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_err,
                'optimizer': self.optimizer.state_dict(),
                'meter': self.meter
        }

    def load_checkpoint(self, filename):
        """Load checkpoint"""
        if os.path.isfile(filename):
            self.logger.info("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.best_err = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(filename))

    def save_checkpoint(self, state, is_best, filename):
        """Save checkpoint"""
        filename = os.path.join(self.model_dir, filename)
        if is_best:
            self.logger.info('=> best model so far, saving...')
            filename = os.path.join(self.model_dir, 'model_best.pth.tar')
        torch.save(state, filename)

    def _get_optimizer(self, lr, momentum, use_adam=opt.use_adam,
                       use_rmsprop=opt.use_rmsprop,
                       use_adagrad=opt.use_adagrad):
        """Get optimizer based off model parameters

        Trainer is defaulted to SGD optimizer if optimizer flags are set to
        False

        """
        parameters = [
        {'params':  self.model.shared_net.features.parameters()},
        {'params': self.model.shared_net.classifier[:6].parameters()},
        # fc8 -> 7th element (index 6) in the Sequential block
        {'params': self.model.shared_net.classifier[6].parameters(), 'lr': 10 * opt.lr}]

        if use_adam:
            return optim.Adam(parameters, lr=lr)

        elif use_rmsprop:
            return optim.RMSprop(parameters, lr=lr, momentum=momentum)

        elif use_adagrad:
            return optim.Adagrad(parameters, lr=lr, momentum=momentum)

        else:
            return optim.SGD(parameters, lr=lr, momentum=momentum)

    def _set_cuda(self):
        """Set computing device to available cuda devices"""
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            computing_device = torch.device("cuda")
            self.logger.debug("CUDA is supported")
            self.logger.debug('PYTORCH Version: {}'.format(torch.__version__))
            self.logger.debug('CUDA Version: {}'.format(torch.version.cuda))
        else:
            computing_device = torch.device("cpu")
        
        return computing_device

def CORAL(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True