# export all images to the db
USAGE = "python export db_path img_path date"

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
		conn = sqlite3.connect(db_file);
		return conn
	except Error as e:
		print(e)

	return None

def insert_image(conn, image):
	""" create a new image in the image table
	:param conn: connection to db
	:param image: image to insert
	:return: image name
	"""
	sql = ''' INSERT INTO images(Filename,
								Gtruth,
								DateCollected,
								Prediction,
								Orientation,
								majorLength,
								minorLength,
								Confidence,
								Height,
								Width) 
			VALUES(?,?,?,?,?,?,?,?,?,?) '''
	cur = conn.cursor()
	cur.execute(sql, image)
	return cur.lastrowid

def loadImages(img_path):
    """	Loads imgs into list of entries
    :param img_path: Abs path to the database.js
    :return: list: List of db values
    """
    curr_db = ""
    with open(img_path, "r") as fconv:
        curr_db = fconv.read()

    db_entries = to_json_format(curr_db)
    return db_entries


def to_json_format(str_db):
    """	Convert str db into python list of db values (json format)
    """
    ind_opbrac = str_db.find("(") + 1
    str_db = str_db[ind_opbrac:]

    ind_brac = str_db.find(")")

    str_db = str_db[:ind_brac]
    
    return json.loads(str_db)

# Execution entrypoint
if __name__ == '__main__':
	
	# check for args
	if len(sys.argv) <= 3:
		print(USAGE)
		exit()
	# declare cl arguments
	db = sys.argv[1]
	img_dat = sys.argv[2]
	date = sys.argv[3]
	
	# get all image files, annotated only for now
	images = loadImages(img_dat)

	# Store in db
	# create connection
	conn = create_connection(db)
	with conn:
		# iterate over the images
		for img in images:
			
			filename = img["url"]
			date_coll = date
			conf = img["prob_proro"] if img["prob_proro"] > img["prob_non_proro"] else img["prob_non_proro"]

			# TODO determine which images to store as annotated.

			image = (filename, img["gtruth"], 
			date_coll, 
			img["pred"], 
			img["orientation"], 
			img["major_axis_length"], 
			img["minor_axis_length"], 
			conf, 
			img["height"], 
			img["width"])

			insert_image(conn, image)