from PIL import Image
import torchvision.transforms as transforms

from utils.constants import Constants as CONST
from utils.config import opt

INPUT_SIZE = opt.input_size if opt.input_size != None else 224
RESCALE_SIZE = INPUT_SIZE
DATA_TRANSFORMS = {
    CONST.TRAIN: transforms.Compose([transforms.Resize(RESCALE_SIZE),
                                     transforms.CenterCrop(INPUT_SIZE),
                                     transforms.RandomAffine(
                                         360,
                                         translate=(0.1, 0.1),
                                         scale=None,
                                         shear=None,
                                         resample=Image.BICUBIC,
                                         fillcolor=0),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                     ]),
    CONST.VAL: transforms.Compose([transforms.Resize(RESCALE_SIZE),
                                   transforms.CenterCrop(INPUT_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                   ]),
    CONST.DEPLOY: transforms.Compose([transforms.Resize(RESCALE_SIZE),
                                      transforms.CenterCrop(INPUT_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                      ])
}
