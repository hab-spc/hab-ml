from __future__ import absolute_import
from pprint import pprint
import os

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
    data_dir = os.path.join(os.path.abspath(os.getcwd()), 'pa4Data')
    seq_len = 100 #Corresponds to the size of a chunk

    # Network
    network = 'lstm'
    model_dir = './experiments/default'
    num_units = 100
    num_layers = 1
    num_outputs = 93
    drop_out = 0

    # Training hyperparameters
    lr = 0.0001
    epochs = 10
    batch_size = 1

    # Optimizer
    use_adam = True
    use_rmsprop = False
    use_adagrad = False

    # Pytorch
    if torch.cuda.is_available():
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    # Training flags
    resume = None
    print_freq = 500
    save_freq = 2
    early_stop = True
    estop_threshold = 3

    # Generate Output flags
    generate_music = True


    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

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