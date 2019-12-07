import pandas
import tensorflow as tf
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
'''
#load original csv file and drop columns, the out put to example csv
df = pandas.read_csv('chicago-taxi-rides-2016/chicago_taxi_trips_2016_01.csv')
print(df.head())
print(df.info())
df = df.drop(columns=['taxi_id', 'trip_start_timestamp', 'trip_end_timestamp', 
'pickup_census_tract', 'dropoff_census_tract', 'payment_type'])
print(df.head())
df.to_csv (r'example_dataframe.csv', index = None, header=True)
'''
df = pandas.read_csv('example_dataframe.csv')
print(df.head())
print(df.info())
y = df.pop('tips')
print(y.head())
print(df.head())

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(df.to_numpy(), y.to_numpy(), epochs=1, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(df.to_numpy(), y.to_numpy())
print('Accuracy: %.2f' % (accuracy*100))