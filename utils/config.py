from __future__ import absolute_import
from pprint import pprint
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch

from utils.constants import *

class Config:
    """Default Configs for training

    After initializing instance of Config, user can import configurations as a
    state dictionary into other files. User can also add additional
    configuration items by initializing them below.

    Example for importing and using `opt`:
    config.py
        >> opt = Config()

    main.py
        >> from config import opt
        >> lr = opt.lr

    NOTE that, config items could be overwriten by passing
    argument `set_config()`. e.g. --voc-data-dir='./data/'

    """

    # Mode
    mode = TRAIN

    # Data
    data_dir = "/data6/lekevin/hab-spc/phytoplankton-db/csv/proro"

    # Network
    arch = 'resnet50'
    model_dir = './experiments/model_best.pth.tar'
    input_size = 112

    # Training hyperparameters
    lr = 0.001
    epochs = 15
    batch_size = 16

    # Optimizer
    use_adam = True
    use_rmsprop = False
    use_adagrad = False

    # Pytorch
    gpu = '2'
    if torch.cuda.is_available():
        num_workers = 4
        pin_memory = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    else:
        num_workers = 0
        pin_memory = False

    # Training flags
    resume = None
    print_freq = 50
    save_freq = 2
    early_stop = True
    estop_threshold = 3
    log2file = False

    # Deploy Hyperparameters
    lab_config = False
    deploy_data = None

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            if k == 'gpu':
                if torch.cuda.is_available():
                    # os.environ["CUDA_VISIBLE_DEVICES"] = v
                    print(v)
            setattr(self, k, v)

        # print('======user config========')
        # pprint(self._state_dict())
        # print('==========end============')

    def _state_dict(self):
        """Return current configuration state

        Allows user to view current state of the configurations

        Example:
        >>  from config import opt
        >> print(opt._state_dict())

        """
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

def set_config(**kwargs):
    """ Set configuration to train/test model

    Able to set configurations dynamically without changing fixed value
    within Config initialization. Keyword arguments in here will overwrite
    preset configurations under `Config()`.

    Example:
    Below is an example for changing the print frequency of the loss and
    accuracy logs.

    >> opt = set_config(print_freq=50) # Default print_freq=10
    >> ...
    >> model, meter = train(trainer=music_trainer, data_loader=data_loader,
                            print_freq=opt.print_freq) # PASSED HERE
    """
    opt._parse(kwargs)
    return opt

opt = Config()