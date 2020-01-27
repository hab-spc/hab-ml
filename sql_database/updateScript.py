#Standard Level Imports
import os
import sys
import sqlite3
from sqlite3 import Error
import datetime
from pathlib import Path

#Project level import
from sql_database.genericconstants import DBConstants as CONST



def updatePreds(updateList):

    """
        Takes in an array of tuples and updates images in 
        the plankton.db database with model prediction 
        probabilities, model name and time of database
        update.
        
        Input - takes in an array of tuples of the following structure:
                (prediction(int), confidence(float), model_name(string), image_filename(string)) 

        Output - None
    """

    # dbPath = os.path.join(Path(__file__).resolve().parents[2],'DB/test.db')
    dbPath = '/data6/phytoplankton-db/hab_in_vitro/hab_service/test.db'

    try:
        conn = sqlite3.connect(dbPath)
        cursor = conn.cursor()
        updateQuery = """ Update {} set {}=?, {}=?, {}=?, {}=? where {}=?"""
        updateQuery = updateQuery.format(CONST.date_table, CONST.PRED, CONST.PROB, CONST.MODEL_NAME, CONST.PRED_TSTAMP, CONST.IMG_FNAME)
        currTime = datetime.datetime.now().strftime("%H:%M:%S")
        columnVals = [each[:-1] + (currTime, each[-1]) for each in updateList]
        cursor.executemany(updateQuery, columnVals)
        conn.commit()
        cursor.close()
    
    except sqlite3.Error as error:
        print("Failed to update table ", error)
    
    finally:
        if (conn):
            conn.close()