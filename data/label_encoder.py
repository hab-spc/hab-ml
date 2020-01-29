"""Encode/decode data"""

# Standard dist imports
import logging
import os

# Third party imports
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Project level imports
from utils.constants import Constants as CONST
from utils.config import opt
from data.d_utils import grab_classes, get_mapping_dict

class HABLblEncoder(LabelEncoder):
    def __init__(self, mode=CONST.DEPLOY, classes_fname=None):
        self.mode = mode
        self.logger = logging.getLogger(__name__)

        if not classes_fname or (self.mode != CONST.DEPLOY):
            classes_fname = os.path.join(opt.model_dir, opt.classes_fname)
            self.logger.debug(f'Classes file not given. Defaulting to {classes_fname}')
            if not os.path.exists(classes_fname):
                raise OSError('File not found. Please double check if it has been '
                              'generated via training')

        self.classes_ = grab_classes(mode=self.mode, filename=classes_fname)

        # fit to classes
        self.fit(self.classes_)

        # grab hab_eval mapping
        self.hab_eval_mapping = get_mapping_dict(original_label_col='c34_workshop',
                                                 mapped_label_col=CONST.HAB_EVAL.upper())

    def hab_transform(self, values):
        assert not isinstance(values, str)
        try:
            encoded = []
            values = np.asarray(values)
            for v in values:
                # if it already exists in the mapping return the value
                if v in set(self.hab_eval_mapping.values()):
                    encoded.append(v)
                else:
                    encoded.append(self.hab_eval_mapping[v])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s" % str(e))

        return np.array(encoded)
