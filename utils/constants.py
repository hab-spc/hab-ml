""" Project Level Constants

Main intention of this is to be able to reduce the complexity of refactoring
'str' based inputs/outputs by initializing them as Project Lvl Constants.

"""
#TODO organize this and make it look readable
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
MODEL_DIR = 'model_dir'
DATA_DIR = 'data_dir'
TRAIN = 'train'; VAL = 'val'; DEPLOY = 'deploy'
IMG = 'images'; LBL = 'labels'
TRANSFORM = 'transform'
INPUT_SIZE = 'input_size'
COLOR = 'color'
PRETRAINED = 'pretrained'
FREEZE = 'freeze'
WEIGHTED = 'weighted_loss'
LABEL = 'label'
ARCH = 'arch'

GPU = 'gpu'
PRINT_FREQ = 'print_freq'
SAVE_FREQ = 'save_freq'
LOG2FILE = 'log2file'

# Deploy Hyperparameters
DEPLOY_DATA = 'deploy_data'
LAB_CONFIG = 'lab_config'
class SPCData():
    USR_LBL = 'user_labels'
    ID = 'image_id'
    IMG = 'images'
    LBL = 'label'
