import pandas as pd
import numpy as np
import pickle
from scipy.signal import butter, lfilter

path = './data/train/subj{}_series{}_data.csv'

data_files = []
b,a = butter(3,1/250.0,btype='lowpass')
for subj in range(1, 12 + 1):
    for series in range(1, 8 + 1):
        print('{}, {}'.format(subj, series))
        df = pd.read_csv(path.format(subj, series))
        df.drop('id', axis=1, inplace=True)
        df = pd.DataFrame(lfilter(b, a, df, axis=0))
        data_files.append(df)

full_df = pd.concat(data_files)
mean = [0] + full_df.mean().tolist()
std = [1] + full_df.std().tolist()
del full_df

print('Saving results')
with open('./data/mean_std.pickle', 'wb') as f:
    pickle.dump((mean, std), f)
