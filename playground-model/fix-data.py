import os
import shutil
import numpy as np 
import pandas as pd 

source_root_dir = "./proro"
dest_root_dir = "data"
data_arr = ["train", "val"]


train_file_path = os.path.join(source_root_dir, "proro_" + "train" + ".csv")
data = pd.read_csv(train_file_path)
classes = data.label.unique()

for each in data_arr:
    temp = dest_root_dir + "/" + each
    for item in classes:
        temp_dir_path = temp + "/" + item
        os.makedirs(temp_dir_path)


print("Done creating directories")

for each in data_arr:
    source_file_path = os.path.join(source_root_dir, "proro_" + each + ".csv")
    data = pd.read_csv(source_file_path)
    sub_dir_path = dest_root_dir + "/" + each
    for row in data.itertuples():
        source_path = row.images
        dest_path = os.path.join(sub_dir_path, row.label + "/" + str(row.Index) + ".jpg")
        shutil.copyfile(source_path, dest_path)

print("Done copying all files")



