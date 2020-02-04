"""Encode/decode data"""

# Standard dist imports
import logging
import os

# Third party imports
from sklearn.preprocessing import LabelEncoder

# Project level imports
from utils.constants import Constants as CONST
from utils.config import opt
from data.d_utils import grab_classes

class HABLblEncoder(LabelEncoder):
    """LabelEncoder object used to transform labels into HAB labels

    LabelEncoder is initialized with popular functions, such as `transform()`,
    `fit_transform()`, etc. for usage.

    """

    def __init__(self, mode=CONST.DEPLOY, classes_fname=None):
        """Initializes HABLblEncoder

        Args:
            mode (str):
            classes_fname (str): Classes filename for grabbing classes
        """

        self.mode = mode
        self.logger = logging.getLogger(__name__)

        if not classes_fname or (self.mode != CONST.DEPLOY):
            classes_fname = os.path.join(opt.model_dir, opt.classes_fname)
            self.logger.debug(f'Classes file not given. Defaulting to {classes_fname}')
            if not os.path.exists(classes_fname):
                raise OSError('File not found. Please double check if it has been '
                              'generated via training')

        self.classes = grab_classes(mode=self.mode, filename=classes_fname)
        self.hab_classes = open(opt.hab_eval_classes, 'r').read().splitlines()
        self.habcls2idx = {cls:idx for idx, cls in enumerate(self.hab_classes)}

        # fit to classes
        self.fit(self.classes)

    def hab_map(self, value):
        """Helper function to map classes to hab classes

        Args:
            value (str): Class name

        Returns:
            str: Mapped Class name

        """
        if value in self.hab_classes:
            return value
        else:
            return 'Other'

    def hab_transform2idx(self, values):
        """Transform class indices into the index of its corresponding HAB class

        Args:
            values (list): List of indices corresponding to current class

        Returns:
            list: List of indices corresponding to new HAB class

        """
        self.fit(self.classes)
        orig_labels = self.inverse_transform(values)
        # switch to hab mapping
        temp_habcls = list(map(self.hab_map, orig_labels))
        return [self.habcls2idx[cls] for cls in temp_habcls]

