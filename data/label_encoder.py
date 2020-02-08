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

    def __init__(self, mode=CONST.DEPLOY):
        """Initializes HABLblEncoder

        Args:
            mode (str):
            classes_fname (str): Classes filename for grabbing classes
        """

        self.mode = mode
        self.logger = logging.getLogger(__name__)

        self.hab_classes = open(opt.hab_eval_classes, 'r').read().splitlines()
        self.habcls2idx = {cls:idx for idx, cls in enumerate(self.hab_classes)}

    def grab_classes(self, data=None, model_dir=None):
        model_dir = model_dir if model_dir != None else opt.model_dir

        if self.mode == CONST.TRAIN or self.mode == CONST.VAL:
            classes_fname = os.path.join(model_dir, '{}_data.info'.format(self.mode))
            df_unique = data[CONST.LBL].unique()

            with open(classes_fname, 'w') as f:
                f.write(str(data[CONST.LBL].value_counts()))

            # gets classes based off dataframe
            self.classes = opt.classes = grab_classes(self.mode, df_unique=df_unique)
        else:
            # gets classes based off the train_data.info that is written during training
            classes_fname = os.path.join(model_dir, 'train_data.info')
            self.classes = opt.classes = grab_classes(self.mode, filename=classes_fname)

        self.num_class = len(self.classes)
        return self.classes, self.num_class

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

