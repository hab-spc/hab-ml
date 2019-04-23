import os

import pandas as pd
import matplotlib.pyplot as plt

# Project Lvl Imports
from prepare_db.create_csv import create_density_csv
from prepare_db.parse_csv import SPCParser

# Module level constants
IMG_TIME_ELAPSE = 3  # Number of hours that images were pulled from
MICRO_COLUMNS = 'micro_proro,micro_total-phyto'.split(',')
IMG_COLUMNS = ['corrected_Non-Prorocentrum', 'corrected_Prorocentrum',
       'clsfier_Prorocentrum', 'clsfier_Non-Prorocentrum']
PRORO_COLUMNS = ['micro_proro', 'corrected_Prorocentrum',
                 'clsfier_Prorocentrum']
PHYTO_COLUMNS = 'image_total-phyto,micro_total-phyto'.split(',')


def retrieve_x_y(columns, df):
    """Retrieve Microscopy (X) and Image (Y) as list types"""
    assert isinstance(columns, list)
    X = df[columns[1]].tolist()
    Y = df[columns[0]].tolist()
    return X, Y

def best_fit(X, Y):
    """Best fit line computation"""
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    return a, b

def plot_results(columns, data):
    """Plot correlation graphs"""
    x, y = retrieve_x_y(columns, data)
    a, b = best_fit(x, y)
    yfit = [a + b * xi for xi in x]
    plt.scatter(x, y)
    plt.xlabel('Density by Microscopy (cells/ml)')
    plt.ylabel('Density by Imaging (images/hour)')
    plt.plot(x, yfit, color='r')
    return data[columns]

if __name__ == '__main__':
    # Load Raw Density Data
    data_dir = '/data6/lekevin/hab-spc'
    fname = os.path.join(data_dir, 'rawdata/Density_data.csv')
    if not os.path.exists(fname):
        print('Creating density data')
        micro_csv = os.path.join(data_dir, "rawdata/Micro_data.csv")
        image_csv = os.path.join(data_dir, "rawdata/SPCImage_data.csv")
        create_density_csv(data_dir, micro_csv, image_csv)

    df = pd.read_csv(fname)
    m, n = df.shape

    # Clean up nan values
    df = df.dropna().reset_index(drop=True)
    print('{} rows dropped ({}-{})'.format(m-df.shape[0], m, df.shape[0]))

    # Convert microscopy estimates to mL
    df[MICRO_COLUMNS] = df[MICRO_COLUMNS].apply(lambda x: x/1000, axis=1)

    # Compute concentration
    df[IMG_COLUMNS] = df[IMG_COLUMNS].apply(SPCParser.get_density_estimate,
                                            axis=1)

    # Plot results using converted dataframe
    # image estimates conversion occur within `plot_results()`
    print('=' * 30 + 'Prorocentrum' + '=' * 30)
    proro_df = plot_results(PRORO_COLUMNS, df)
    print('=' * 30 + 'Total Phytoplankton' + '=' * 30)
    phyto_df = plot_results(PHYTO_COLUMNS, df)
