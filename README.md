# HAB detection with HAB-ML 

The code here is for training, validating, and deploying HAB classification models, using convolutional neural networks in PyTorch.

## What is HAB-ML?

HAB-ML (Harmful Algae Bloom Machine Learning) is a machine learning library for detecting the algae species from images collected by the Scripps Plankton Camera (SPC).
There are a total of 46 species of interest, that need to be detected from the SPC database. For all HAB-ML models, they are pretrained off of ImageNet
and fine tuned for this task.

## Our task
The specific task here is to identify a harmful algae species at the species taxonomy level, i.e. given an image, return the predicted species classification.

### Experiment
The following method is used to generate a pretrain weight for our model. 
1. Unsupervised Feature Learning via Non-parameteric Instance Discrimination. [(arxiv)](https://arxiv.org/pdf/1805.01978.pdf) [(submodule code)](https://github.com/zhirongw/lemniscate.pytorch) This is done in <code>main_instance.py</code>.

### Getting Started
The approach here is to add a classifier on top of the pre-trained CNN model and fine tune the model parameters by training on our domain specific data, i.e., phytoplankton images.
This is done in <code>main.py</code>.

#### System Requirement
1. Python 3.5 or higher
2. PyTorch 1.0.1 or higher
3. Python libraries: pandas, scikit-learn, scikit-image, matplotlib, numpy, lxml
4. Example: Create a python environment named `hab_env` and install required libraries using `pip`:
    - `virtualenv hab_env`
    - `source hab_env/bin/activate`
    - `pip install --user -r requirements.txt`
5. 12GB GPU (recommended)

It is recommended to run the program on GPU with 12 GB memory. Training (fine tuning) the model on a GPU, with ~10k images in training data, takes ~1 hour for 5 or 6 epochs depending on the hyperparameters used.
The scoring can be done on CPU, but running on GPU is ~10-30x faster than on CPU

#### Download required files:
1. Copy the files and directories in this repo into your work directory.
2. Download the pre-trained ResNet models into the `model` directory of your work directory. 
    - Run `python download_pretrained.py` from the `model` directory.
    
    
#### Execute the job

##### Run a saved model
To run a saved model in `main.py`, include deploy type arguments with the script call.
This will run the fine-tuned model saved in the specified `model` directory to evaluate a batch of data.
An example is given below, where we are running the `20_class_run` model version on a dataset collected in 2018.
```
deploy_data=/data6/phytoplankton-db/csv/field_2018.csv
model_dir=/data6/yuanzhouyuan/hab/hab-ml/experiments/20_class_run/
python main.py --mode deploy --batch_size 16 --deploy_data $deploy_data --model_dir $model_dir
```

##### Train and validate the model

###### 1. Run the main.py
You can run it with different arguments, such as --mode, --model_dir, --epochs, --lr. 
###### 2. Answer the input questions:
1. `Enter training set date (ex.20190708) : `
    - To answer this question, type the date. Then, the code will set 1data_dir1 to `/data6/plankton_test_db_new/data/$(date)/` directory to search for train.log and val.log. [The current workshop data folder is stored to date 0]
2. `Do you want to load existed checkpoint ? (y/n)`
    - If you type `y`, then, it will set `resume` to True. And when it was loading checkpoint in Trainer.py, you will be asked to query the sqlite database in `/data6/plankton_test_db_new/model/model.db`.
    - If you type `n`, it will just load the ImageNet pre-trained weights.
3. `Do you want to save model to sql database? (y/n)`
    - If you type `y`,  It will ask you to enter today's date. Then, it will set `model_dir` into `/data6/plankton_test_db_new/model/$(date)/$(time)/`, and save the model info into it. 
    - If you type `n`, it will save model to your `model_dir` you specified in argument. If you did not specified `model_dir` in argument, it will save to the default directory.

Sample 1: In this sample we will i) select a new training set ii) save the model to the sql database
```
(hab) zhy185@gpu2:/data6/yuanzhouyuan/hab/hab-ml$ python main.py --mode TRAIN --model_dir ./experiments/proro_run/ --epochs 0

Enter training set date (ex.20190708) :
>>> 0
# now the data_dir is /data6/plankton_test_db_new/data/0

Do you want to load existed checkpoint ? (y/n)
>>> n
Do you want to save model to sql database? (y/n)
>>> y
Enter today date (ex.20190708) :
>>> 20190802
#now model_dir will be saved to /data6/plankton_test_db_new/model/20190802/$(current_time)/

#Inserting current model to sql database
Any Addtional Comment to this model?
>>> just for testing, better not to use it
#the additional comment of this specific model is added to sqlite

```

Sample 2: In this example we will i) load an existing checkpoint (6) and ii) save the model to the sql database
```
(hab) zhy185@gpu2:/data6/yuanzhouyuan/hab/hab-ml$ python main.py --mode TRAIN --model_dir ./experiments/proro_run/ --epochs 0

Enter training set date (ex.20190708) :
>>> 0
# data_dir will be /data6/plankton_test_db_new/data/0
Do you want to load existed checkpoint ? (y/n)
>>> y
Do you want to save model to sql database? (y/n)
>>> y
Enter today date (ex.20190708) :
>>> 20190802
#model_dir will be /data6/plankton_test_db_new/model/20190802/$(current_time)/

# search for checkpoint
Enter query command (ex. SELECT * FROM models WHERE train_acc > 90) :
>>> SELECT * FROM models

#below are checkpoints info
#[ID, model name, model path, frezzing layers number, uselss variable, learning rate, batch size, best train acc, best test acc, epochs, date, class number, additional comment]
(1, 'resnet50', '/data6/plankton_test_db_new/model/20190730/1564555883/', 0, 'n', 0.001, 16, 38.833333333333336, 21.15506329113924, 1, 20190730, 24, 'just for testing, better not to use it')
(2, 'resnet50', '/data6/plankton_test_db_new/model/20190730/1564556191/', 0, 'n', 0.001, 16, 57.06944444444444, 30.799050632911392, 1, 20190730, 24, 'used for testing resume, no need to use it')
(3, 'resnet50', '/data6/plankton_test_db_new/model/20190730/1564556466/', 0, 'n', 0.001, 16, 0.0, 0.0, 15, 20190730, 24, 'used for testing VAL mode, no need to use it')
(4, 'resnet50', '/data6/plankton_test_db_new/model/20190730/1564557013/', 0, 'n', 0.001, 16, 89.80555555555556, 68.44145569620254, 15, 20190730, 24, 'testing without summary_state report')
(5, 'resnet50', '/data6/plankton_test_db_new/model/20190731/1564629304/', 0, 'n', 0.001, 16, 94.18402777777777, 70.57773109243698, 15, 20190731, 23, 'first try with 23 classes, performance is not good')
(6, 'resnet50', '/data6/plankton_test_db_new/model/20190802/1564786478/', 0, 'n', 0.001, 16, 0.0, 0.0, 0, 20190802, 14, 'just for testing, better not to use it')

Enter the ID number of the model:
>>> 6

The model you find is:
[(6, 'resnet50', '/data6/plankton_test_db_new/model/20190802/1564786478/', 0, 'n', 0.001, 16, 0.0, 0.0, 0, 20190802, 14, 'just for testing, better not to use it')]

# checkpoint '/data6/plankton_test_db_new/model/20190802/1564786478/model_best.pth.tar' will be loaded

#After training, i can enter some additional comment to the newly created model
Any Addtional Comment to this model?
>>> just for testing, don't use

```

Sample 3: In this example, we will not be loading an existing checkpoint or saving the the model
```
(hab) zhy185@gpu2:/data6/yuanzhouyuan/hab/hab-ml$ python main.py --mode TRAIN --model_dir ./experiments/proro_run/ --epochs 0

Enter training set date (ex.20190708) :
>>> 0
# data_dir will be /data6/plankton_test_db_new/data/0
Do you want to load existed checkpoint ? (y/n)
>>> n
Do you want to save model to sql database? (y/n)
>>> n
#model_dir will be the one specified in the arguments. The model_dir will be ./experiments/proro_run/

```

## Benchmark
We fine tuned the ResNet50 model with different model size and hyperparameters and tested on one of our validation sets.
The saved model, `model`, uses the highlighted parameters below.

| Model | input_size | batch_size | learning_rate | epochs | training_time | Accuracy (val) |
|-------|------------|------------|---------------|--------|---------------|----------------|
|       |            |            |               |        |               |                |
|       |            |            |               |        |               |                |
|       |            |            |               |        |               |                |

Directory Structure
------------

The directory structure of the HAB-ML project looks like this: 

```
├── README.md          <- The top-level README for developers using this project.
├── data               <- Scripts to download or generate data
│   ├── dataloader.py
│   ├── d_utils.py
│   └── prepare_db.py
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   ├── download_pretrained.py
│   ├── layers.py
│   └── resnet.py
├── spc                <- SPICI module to download and upload SPC images
│
├── sql database       <- Scripts to get and set data in sql database
│
├── utils              <- Miscellaneous scripts
│   ├── config.py
│   ├── constants.py
│   ├── eval_utils.py
│   ├── logger.py
│   └── model_sql.py
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── main.py            <- Script to train models and then use trained models to make
│                         predictions
│
└── trainer.py         <- Module to handle model training
```


