import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

micro_Data = pd.read_csv("rawdata/Micro_data.csv")
image_Data = pd.read_csv("rawdata/Image_data.csv", skiprows=[1, 2, 3])

dates_micro = list(micro_Data['Sample ID\n\nYYYYMMDD'])
for i in range(len(dates_micro)):
    temp = str(dates_micro[i])
    date_form = temp[0:4] + '-' + temp[4:6] + '-' + temp[6:8]
    dates_micro[i] = date_form
proro_micro = list(micro_Data['Prorocentrum micans (Cells/L)'])  # in cells per liter
proro_density_micro = [m / 1000 for m in proro_micro]  # in cells per ml

dates_image = list(image_Data['Start Date & Time'])
for i in range(len(dates_image)):
    temp = str(dates_image[i])
    date_form = temp[0:4] + '-' + temp[5:7] + '-' + temp[8:10]
    dates_image[i] = date_form
time_elapse = list(image_Data['Time elapsed '])
proro_count = list(image_Data['# Proro detected'])
proro_density_image = [proro_count[i] / (int(time_elapse[i][0]) * 3600) for i in range(len(dates_image))]

# Synchronizing the Dates

density_micro = []  # in number of cells/litre
density_image = []  # in number of images/second
common_dates = []

for i in range(len(dates_micro)):
    for j in range(len(dates_image)):

        if dates_micro[i] == dates_image[j]:

            if (str(proro_density_image[j]) == 'nan'):

                print('date of recording', dates_micro[i])
                print('Number of Prorocentrums counted', proro_density_image[j])

            else:

                density_micro.append(proro_density_micro[i])
                density_image.append(proro_density_image[j])
                common_dates.append(dates_micro[i])

df = pd.DataFrame(list(zip(density_micro, density_image)), columns=['Micro', 'Image'])
X = df.iloc[:, 0].values.reshape(-1, 1)
Y = df.iloc[:, 1].values.reshape(-1, 1)
plt.scatter(X, Y)
plt.xlabel('Density by Microscopy (cells/ml)')
plt.ylabel('Density by Imaging (images/second)')
plt.show()

# Generating a CSV File with density estimates and their Date of recording
den_n_dates = pd.DataFrame({'Date': common_dates,
                            'Microscopy Density Estimate': density_micro,
                            'Imaging Density Estimate': density_image},
                            columns=['Date', 'Microscopy Density Estimate', 'Imaging Density Estimate'])

den_n_dates.to_csv('rawdata/consisedata.csv')