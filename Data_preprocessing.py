#import pandas
'''
import numpy as np


a = np.full((2, 2),2)
b = np.full((2, 2),1)
print(a)
print(b)
np.save('test3.npy', a)    # .npy extension is added if not given
d = np.load('test3.npy')
print(d)
'''
import pandas as pd
from sklearn import preprocessing
import numpy as np
df = pd.read_csv('chicago-taxi-rides-2016/chicago_taxi_trips_2016_01.csv')
print(df.head())
print(df.info())
df = df.drop(columns=['taxi_id', 'trip_start_timestamp', 'trip_end_timestamp', 
'pickup_census_tract', 'dropoff_census_tract', 'payment_type'])
print(df.head())
y = df.pop('tips')
df = df.to_numpy()
y = y.to_numpy()
where_are_NaNs = np.isnan(df)
df[where_are_NaNs] = 0

where_are_NaNs = np.isnan(y)
y[where_are_NaNs] = 0


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(x_scaled)
print(df)
print(y)