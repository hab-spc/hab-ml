from collections import OrderedDict

class GenericConstants:
    RAW_DATA = 'raw'
    PROCESSED_DATA = 'processed'
    CURRENT_ENV = 'current_environment'
    DEV_ENV = 'dev_mac'
    PROD_ENV = 'prod_env'
    LIVIS = 'livis'

class DBConstants:
    date_table = 'date_sampled'

    pre = 'image_'
    # Image Info
    IMG_FNAME = pre + 'filename'
    IMG_ID = pre + 'id'
    IMG_TSTAMP = pre + 'timestamp'
    IMG_DATE = pre + 'date'
    IMG_TIME = pre + 'time'
    IMG_FSIZE = pre + 'file_size'
    ECCENTRICITY = pre + 'eccentricity'
    ORIENT = pre + 'orientation'
    MJR_LEN = pre + 'major_axis_length'
    MIN_LEN = pre + 'minor_axis_length'
    HEIGHT = pre + 'height'
    WIDTH = pre + 'width'
    SOLIDITY = pre + 'solidity'
    ASPT_RATIO = pre + 'aspect_ratio'
    EST_VOL = pre + 'estimated_volume'
    AREA = pre + 'area'

    # Machine Learning Info
    pre = 'ml_'
    MODEL_NAME = pre + 'model_name'
    USR_LBLS = pre + 'user_labels'
    PRED = pre + 'prediction'
    PROB = pre + 'probability'
    PRED_TSTAMP = pre + 'prediction_timestamp'

    # Annotation Info
    pre = 'annot_'
    IMG_STATUS = pre + 'image_status'
    IMG_TAG = pre + 'image_tags'
    ML_LBL = pre + 'machine_label'
    HMN_LBL = pre + 'human_label'

    def _state_dict(self, type='image'):
        """Return current configuration state
        Allows user to view current state of the configurations
        Example:
        >>  from genericconstants import DBConstants
        >>  db = DBConstants()
        >>  image_info_to_update = db._state_dict(type='image').values()
        Out: odict_values(['image_filename', 'image_id', ..., 'image_area'])
        """
        return OrderedDict({k: getattr(self, k) for k, v in DBConstants().__dict__.items() \
                            if not k.startswith('_') and v.startswith(type)})

    @property
    def image_fields(self):
        return DBConstants()._state_dict(type='image').values()

    @property
    def ml_fields(self):
        return DBConstants()._state_dict(type='ml').values()

    @property
    def annot_fields(self):
        return DBConstants()._state_dict(type='annot').values()
