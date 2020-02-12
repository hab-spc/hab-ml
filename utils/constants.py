""" Project Level Constants

Main intention of this is to be able to reduce the complexity of refactoring
'str' based inputs/outputs by initializing them as Project Lvl Constants.

"""


class Constants():
    LAYER_SPECS = 'layer_specs'
    ACTIVATION = 'activation'
    BATCH = 'batch_size'
    EPOCHS = 'epochs'
    EARLY_STOP = 'early_stop'
    ESTOP_THRESH = 'early_stop_epoch'
    WEIGHT_DECAY = 'weight_decay'
    MOMENTUM = 'momentum'
    GAMMA = 'momentum_gamma'
    LR = 'lr'
    RESUME = 'resume'
    START_EPOCH = 'start_epoch'
    BEST_LOSS = 'best_loss'
    MODE = 'mode'
    INTERACTIVE = 'interactive'
    MODEL_DIR = 'model_dir'
    DATA_DIR = 'data_dir'
    TRAIN = 'train';
    VAL = 'val';
    DEPLOY = 'deploy'
    TRANSFORM = 'transform'
    INPUT_SIZE = 'input_size'
    COLOR = 'color'
    PRETRAINED = 'pretrained'
    FREEZED_LAYERS = 'freezed_layers'
    WEIGHTED_LOSS = 'weighted_loss'
    ARCH = 'arch'
    SAVE_MODEL_DB = 'save_model_db'

    GPU = 'gpu'
    PRINT_FREQ = 'print_freq'
    SAVE_FREQ = 'save_freq'
    LOG2FILE = 'log2file'
    LOGGING_LVL = 'logging_level'

    # Deploy Hyperparameters
    DEPLOY_DATA = 'deploy_data'
    LAB_CONFIG = 'lab_config'
    HAB_EVAL = 'hab_eval'

    # Data
    IMG = 'images'
    LBL = 'label'
    ID = 'image_id'
    USR_LBL = 'user_labels'

    # Machine Learning Info
    pre = 'ml_'
    MODEL_NAME = pre + 'model_name'
    USR_LBLS = pre + 'user_labels'
    PRED = pre + 'prediction'
    HAB_PRED = pre + 'hab_prediction'
    PROB = pre + 'probability'
    PRED_TSTAMP = pre + 'prediction_timestamp'
    
    # Instance NCE
    LOW_DIM = 'low_dim'
    NCE_K = 'nce_k'
    NCE_T = 'nce_t'
    NCE_M = 'nce_m'

"""TO BE DEPRECATED"""
class SPCConstants():
    USR_LBL = 'user_labels'
    ID = 'image_id'
    IMG = 'images'
    LBL = 'label'
