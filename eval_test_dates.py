import os
import multiprocessing as mp
model_dir = [
    # 'experiments/vgg16_pretrained_c34_workshop2019',
    # 'experiments/resnet18_pretrained_c10_workshop2019',
    # 'experiments/resnet18_lab_coral',
    # 'experiments/resnet18_pretrained_c34_workshop2019_2',
    # 'experiments/default',
    # './experiments/resnet_coral_c34_workshop2019/',
    # './experiments/resnet_no_coral_c34_workshop2019/',
    # 'experiments/resnet18_pretrained_c22_workshop2019',
    # 'experiments/resnet18_pretrained_c51_raw_workshop2019'
    'experiments/resnet18_pretrained_c34_workshop2019'
]
test_sets = [
    #PIER
    'data/DB/csv/hab_in_situ_20190523.csv',
    'data/DB/csv/hab_in_situ_20190528.csv',
    #LAB
    'data/DB/csv/hab_in_vitro_20190523.csv',
    'data/DB/csv/hab_in_vitro_20190528.csv',
    'data/DB/csv/hab_in_vitro_20190530.csv',
    'data/DB/csv/hab_in_vitro_20190603.csv',
    'data/DB/csv/hab_in_vitro_20190610.csv',
    'data/DB/csv/hab_in_vitro_20190624.csv',
    'data/DB/csv/hab_in_vitro_20190729.csv',
    'data/DB/csv/hab_in_vitro_20190815.csv',
    'data/DB/csv/hab_in_vitro_20190826.csv',
    'data/DB/csv/hab_in_vitro_20190930.csv',
    'data/DB/csv/hab_in_vitro_20191007.csv'
]
for model in model_dir:
    for test in test_sets:
        cmd = "python main.py --mode deploy --model_dir {} --deploy_data " \
              "{} --batch_size 16 --arch resnet18 --gpu 0 --logging_level=10 " \
              "--hab_eval".format(model, test)
        # cmd = "python coral_main.py  --mode deploy " \
        #       "--model_dir {} --deploy_data {} --arch " \
        #       "resnet18 --coral".format(model,test)
        os.system(cmd)
