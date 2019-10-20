# HAB ML Train and Val

The code here is for training, validating, and deploying hab images.



## Functionality

The code aims to help you to keep track of different models and to be able to continue the training process easily. 

## How to train and validation

### 0. Download
Run 'git clone' to download the code. The code should work in any directory you want in svcl sever. 

### 1. Run the main.py
You can run it with different arguments, such as --mode, --model_dir, --epochs, --lr. 
### 2. Answer the questions:
1. 'Enter training set date (ex.20190708) : '
    - To answer this question, type the date. Then, the code will set data_dir to '/data6/plankton_test_db_new/data/$(date)/' directory to search for train.log and val.log. [The current workshop data folder is stored to date 0]
2. 'Do you want to load existed checkpoint ? (y/n)'
    - If you type 'y', then, it will set 'resume' to True. And when it was loading checkpoint in Trainer.py, you will be asked to query the sqlite database in '/data6/plankton_test_db_new/model/model.db'.
    - If you type 'n', it will just load the ImageNet pre-trained weights.
3. 'Do you want to save model to sql database? (y/n)'
    - If you type 'y',  It will ask you to enter today's date. Then, it will set model_dir into ''/data6/plankton_test_db_new/model/$(date)/$(time)/', and save the model info into it. 
    - If you type 'n', it will save model to your model_dir you specified in argument. If you did not specified model_dir in argument, it will save to the default directory.

### Sample 1

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

### Sample 2

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

### Sample 3
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

## How to Deploy
To run with mode deploy, no need to enter any user input. The code will go through the model_dir/train_data.info to find the class info. 

```
deploy_data=/data6/phytoplankton-db/csv/field_2018.csv
model_dir=/data6/yuanzhouyuan/hab/hab-ml/experiments/20_class_run/
python main.py --mode deploy --batch_size 16 --deploy_data $deploy_data --model_dir $model_dir
```


## Storage System

- /data6/plankton_test_db_new/
    - data
        - $(date)
            - train.csv
            - val.csv
    - model
        - model.db
        - $(date)
            - model_created_time
                - model_best.pth.tar (best weights which generate lowest loss in validation data)
                - train.log (train/val size, and performance matrix)
                - train_data.info (classes counts in train.csv)
                - val_data.info (classes counts in val.csv)
                - figs
                    - confusion.png (confusion matrix of the validation data with lowest loss)
                    - loss.png (train and val loss over epochs)
                    
