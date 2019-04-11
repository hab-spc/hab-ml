import os

import pandas as pd
import matplotlib.pyplot as plt

# Project Lvl Imports
from prepare_db.create_csv import create_density_csv

# Module level constants
IMG_TIME_ELAPSE = 3  # Number of hours that images were pulled from
MICRO_COLUMNS = 'micro_proro,micro_total-phyto'.split(',')
IMG_COLUMNS = 'image_proro,image_non-proro,image_total-phyto'.split(',')
PRORO_COLUMNS = 'image_proro,micro_proro'.split(',')
PHYTO_COLUMNS = 'image_total-phyto,micro_total-phyto'.split(',')

def compute_time_density(metric, hour=False):
    """Convert image count into rate of cell/time"""
    if hour:
        metric /= 60
        metric = metric/IMG_TIME_ELAPSE
    else:
        metric = metric/(IMG_TIME_ELAPSE*60)
    return metric

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

def plot_results(columns, df):
    """ Plot results w.r.t to several time metrics

    Image density conversions occur within here, since there are multiple
    time conversions to conduct.

    Args:
        columns: (list) Image and Microscopy columns for given species
        df: (pd.Dataframe) Raw density data

    Returns:
        dict: Structure containing the transformed dataframe to provide
            validation according to the time conversion

    """
    times = [1, 5, 15, 30, 60, 180] # m, m, m, h, h
    df_for_val = {} # Store transformed estimates df according to its time
    n_rows, n_cols = 1, 6
    plt_width, plt_height = 5, 5
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*plt_width, n_rows*plt_height))
    for ii, t in enumerate(times):
        df_ = df.copy()

        # Compute time density
        hour = True if ii > 3 else False
        metric = compute_time_density(t, hour)
        df_[IMG_COLUMNS] *= metric

        # Save transformed estimates
        df_for_val[t] = df_[columns]

        # Plot correlation graph and best fit line
        X, Y = retrieve_x_y(columns, df_)
        a, b = best_fit(X, Y)
        yfit = [a + b * xi for xi in X]
        ax[ii%6].scatter(X, Y)
        ax[ii%6].set_xlabel('Density by Microscopy (cells/ml)')
        ax[ii%6].set_ylabel('Density by Imaging (images/hour)')
        ax[ii%6].plot(X, yfit, color='r')
    return df_for_val

if __name__ == '__main__':
    # Load Raw Density Data
    data_dir = '/data6/lekevin/hab-spc'
    fname = os.path.join(data_dir, 'rawdata/Density_data.csv')
    if not os.path.exists(fname):
        print('Creating density data')
        create_density_csv()
    df = pd.read_csv(fname)
    m, n = df.shape

    # Clean up nan values
    df = df.dropna()
    print('{} rows dropped ({}-{})'.format(m-df.shape[0], m, df.shape[0]))

    # Convert microscopy estimates to mL
    df[MICRO_COLUMNS] = df[MICRO_COLUMNS].apply(lambda x: x/1000, axis=1)

    # Plot results using converted dataframe
    # image estimates conversion occur within `plot_results()`
    print('=' * 30 + 'Prorocentrum' + '=' * 30)
    proro_df = plot_results(PRORO_COLUMNS, df)
    print('=' * 30 + 'Total Phytoplankton' + '=' * 30)
    phyto_df = plot_results(PHYTO_COLUMNS, df)
