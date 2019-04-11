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


# create_proro_csv()

def create_density_csv():
    data_dir = '/data6/lekevin/hab-spc'
    micro_Data = pd.read_csv(os.path.join(data_dir, "rawdata/Micro_data.csv"))
    #TODO prediction counts should come from proro_<PARTITION>.csv files
    image_Data = pd.read_csv(os.path.join(data_dir, "rawdata/Image_data.csv"),
                             skiprows=[1, 2, 3])

    # Process Image_data to merge with microscopy data
    image_Data = image_Data[['Start Date & Time', 'End Date & Time',
                             '# Non-Proro detected', '# Proro detected']]
    image_Data['Date'] = image_Data['Start Date & Time'].str.split(' ').str[0]
    image_Data['Date'] = pd.to_datetime(image_Data['Date'])
    image_Data['Date'] = image_Data['Date'].dt.strftime('%Y-%m-%d')

    # Process Microscopy_data
    CSV_COLUMNS = 'Date,Prorocentrum micans (Cells/L),Total Phytoplankton (Cells/L)'
    micro_Data = micro_Data.rename(columns={'Date\n\nmm/dd/yy': "Date"}, index=str)
    micro_Data['Date'] = pd.to_datetime(micro_Data['Date']).dt.strftime('%Y-%m-%d')
    micro_Data = micro_Data[CSV_COLUMNS.split(',')]

    # Merge two data types
    density_Data = micro_Data.merge(image_Data, on='Date')

    # Rename columns for simplicity
    rename_dict = {'Prorocentrum micans (Cells/L)': 'micro_proro',
                   'Total Phytoplankton (Cells/L)': 'micro_total-phyto',
                   'Start Date & Time': 'image_start-datetime',
                   'End Date & Time': 'image_end-datetime',
                   '# Non-Proro detected': 'image_non-proro',
                   '# Proro detected': 'image_proro'}
    density_Data = density_Data.rename(columns=rename_dict, index=str)

    # Compute total phytoplankton
    density_Data['image_total-phyto'] = density_Data['image_non-proro'] + \
                                        density_Data['image_proro']

    # Save as raw data
    fname = os.path.join(data_dir, 'rawdata/Density_data.csv')
    density_Data.to_csv(fname, index=False)



create_density_csv()


