# create a training set from annotated images
USAGE = "python export db_path date [query]"

import shutil, os
import sys
import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
import json

def create_connection(db_file):
    """ create a db connection to database specified
        by db_file
    :param db_file: database file path
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None

def select_all_images(conn):
    """    Query all rows in the images tables
    :param conn: the Connection object
    :return: list of rows
    """
    cur = conn.cursor()
    cursor = cur.execute("SELECT * FROM images")
    names = [x[0] for x in cursor.description]
    return [cur.fetchall(), names]


def select_image_by_query(conn, query):
    """    Query image by species
    :param conn: the Connection Object
    :param query: manual query
    :return: all rows of the species
    """
    cur = conn.cursor()
    if query == "":
        cursor = cur.execute("SELECT * from images")
    else:
        cursor = cur.execute(query)
    names = [x[0] for x in cursor.description]
    return [cur.fetchall(), names]

def create_csv (df,date):
    """ create train and val csv files given user input
    :param df: pandas dataframe with image details
    :param date: date on which images were taken
    """
    df = df.rename({'Filename': 'images', 'GTruth': 'label'}, axis='columns')
    pre_path = '/data6/phytoplankton-db/hab_in_vitro/images/'+date+'/001/00000_static_html/static/'
    df['images'] = df['images'].apply(lambda x: "{}{}{}".format(pre_path, x[:-5],'.jpeg'))
    acc_arr = (df['label']==df['Prediction']).values.tolist()
    pred_acc = 100*acc_arr.count(True)/(len(acc_arr))
    print('The prediction acc is: '+str(pred_acc)+'%.')
    df = df[['images', 'label']]
    
    #any class over 24 counts as other
    #df.loc[(df['label'] > 23) ,'label'] = 23
    
    freq = df['label'].value_counts()
    print('Classes Counts table: \nclass|counts \n ----------')
    print(freq)

    print('DataFrame constrcuted.')
    stop = False
    while stop == False:
        nums = input('Enter how many images you want each class to have for training. \n (ex. "1," means all class have 1 image for training);\n "1,2," means first class has 1 image, second class has 2 images for training)\n')
        nums = nums.replace(" ", "")
        nums = nums.split(',')
        nums = [x for x in nums if x]
        train_dict = {}
        classes = df['label'].unique()
        print(classes)
        if len(nums) == 1:
            for i in classes:
                train_dict[i] = int(nums[0])
        elif len(nums) == len(classes):
            for i in range(len(classes)):
                train_dict[classes[i]] = int(nums[i])
        else:
            print('class number not match')
            stop = False
            continue
        print('Numbers of Images you select for each class for training: ')
        print(train_dict)
        y_n = input('Are you sure these are the number you want? (y/n): (ex. y) \n')
        if y_n == 'y':
            stop = True
        else:
            stop = False

    train_df = pd.DataFrame()
    for i in classes:
        temp = df.loc[df['label'] == i][:train_dict[i]]
        train_df = train_df.append(temp)
        temp = temp.index
        df = df.drop(temp)

    print('Train and Test Dataframe contructed')
    print('Train Dataframe Class Counts table')
    print(train_df['label'].value_counts())
    print('VAL Dataframe Class Counts table')
    print(df['label'].value_counts())

    #change dc_path
    db_path = '/data6/plankton_test_db_new'
    if not os.path.isdir(db_path+'/data/'+ date):
        os.mkdir(db_path+'/data/'+ date)

    train_path = db_path+'/data/'+date+'/train.csv'
    train_df.to_csv(train_path)
    print('Train Dataframe is stored to '+train_path)

    test_path = db_path+'/data/'+date+'/val.csv'
    df.to_csv(test_path)
    print('Test Dataframe is stored to '+test_path)

    info_path = db_path+'/data/'+date+'/info.txt'
    print('Info is stored to '+ db_path+'/data/'+date+'/info.txt')
    f = open(info_path, 'w')
    sys.stdout = f
    print('The prediction acc is: '+str(pred_acc)+'%.')
    print('Train and Test Dataframe contructed')
    print('Train Dataframe Class Counts table')
    print(train_df['label'].value_counts())
    print('VAL Dataframe Class Counts table')
    print(df['label'].value_counts())
    f.close()


# Execution entrypoint
if __name__ == '__main__':
    # check for args
    if len(sys.argv) <= 2:
        print(USAGE)
        exit()
    # declare cl arguments
    db = sys.argv[1]
    date = sys.argv[2]
    
    # get image files
    # create connection
    conn = create_connection(db)
    with conn:
        if len(sys.argv) == 4:
            images = select_image_by_query(conn, sys.argv[3])
        else:
            images = select_image_by_query(conn, "")
            
    # store images to new csv
    df = pd.DataFrame(images[0], columns =images[1])
    create_csv(df,date)

