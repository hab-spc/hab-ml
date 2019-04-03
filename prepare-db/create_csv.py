"""Create CSV file to hold image meta data

Main usages:
- [TEMP] Generate csv files of in situ images retrieved from the SPC database
- Generate csv files of in vitro images from the lab
- - - images given in year-month-day format, so need to process all images
at once

"""
# Standard dist imports
import glob
import os
import re

# Third party imports
import pandas as pd

# Project level imports

# Module level constants

DATA_DIR = '/data6/lekevin/hab-spc/phytoplankton-db'

def create_proro_csv():
    proro_dir = [f for f in os.listdir(DATA_DIR) if 'proro' in f]

create_proro_csv()