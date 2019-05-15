"""Trainer"""
# Standard dist imports
import logging
import os

# Third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Project level imports
from utils.config import opt
from utils.constants import *
from utils.eval_utils import get_meter

# Module level constants

class Trainer(object):
    """Trains Model

    Trainer is essentially a wrapper around the model, optimizer,
    and criterion. It assists with choosing which optimizer to use through
    flags, typecasting variables to cuda, and loading/saving checkpoints.

    """
    def __init__(self, model, model_dir=opt.model_dir, mode=TRAIN,
                 resume=opt.resume, lr=opt.lr):
        """ Initialize Trainer

        Args:
            model: (MusicRNN) Model
            model_dir: (str) Path to the saved model directory
            mode: (str) Train or Test
            resume: (str) Path to pretrained model

        """
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('trainer')
        self.computing_device = self._set_cuda()

        self.model = model.to(self.computing_device)
        print("Model on CUDA?", next(self.model.parameters()).is_cuda)
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.start_epoch = 0
        self.best_err = np.inf

        self.optimizer = self._get_optimizer(lr=lr)
        # Defaulted to CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()

        # meter
        self.meter = {TRAIN: get_meter(), VAL: get_meter()}

        # Load checkpoint
        if resume or mode != TRAIN:
            if resume:
                fn = resume
            else:
                fn = os.path.join(self.model_dir, 'model_best.pth.tar')
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
            print("=> loading checkpoint '{}'".format(filename))
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

    def _get_optimizer(self, lr, use_adam=opt.use_adam,
                       use_rmsprop=opt.use_rmsprop,
                       use_adagrad=opt.use_adagrad):
        """Get optimizer based off model parameters

        Trainer is defaulted to SGD optimizer if optimizer flags are set to
        False

        """
        if use_adam:
            return optim.Adam(self.model.parameters(), lr=lr)

        elif use_rmsprop:
            return optim.RMSprop(self.model.parameters(), lr=lr)

        elif use_adagrad:
            return optim.Adagrad(self.model.parameters(), lr=lr)

        else:
            return optim.SGD(self.model.parameters(), lr=lr)

    def _set_cuda(self):
        """Set computing device to available cuda devices"""
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            computing_device = torch.device("cuda")
            print("CUDA is supported")
            self.logger.debug('PYTORCH Version: {}'.format(torch.__version__))
            self.logger.debug('CUDA Version: {}'.format(torch.version.cuda))
        else:
            computing_device = torch.device("cpu")
        
        return computing_device