import pandas
from Neural_Network import Neural_Network
import numpy as np
from sklearn import preprocessing
'''
df = pandas.read_csv('example_dataframe.csv')
df = df.head()
df.to_csv (r'test.csv', index = None, header=True)
'''
df = pandas.read_csv('example_dataframe.csv')
#print(df)
#df = df.head()
y = df.pop('tips')
df = df.to_numpy()
y = y.to_numpy()
where_are_NaNs = np.isnan(df)
df[where_are_NaNs] = 0

where_are_NaNs = np.isnan(y)
y[where_are_NaNs] = 0

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df = pandas.DataFrame(x_scaled)
df = df.to_numpy()

WeightInput = np.random.rand(13, 1)
WeightHiddenOne = np.random.rand(13, 11)
WeightHiddenTwo = np.random.rand(11, 8)
WeightOutput = np.random.rand(8, 1)
#print("weihtoutppppput")
#print(WeightOutput)

nn = Neural_Network(WeightHiddenOne, WeightHiddenTwo, WeightOutput)
nn.LoadFromFile()
nn.feed_forward(df)
y = np.reshape(y, (-1, 1))
nn.printShapes()

print("------------------------------------------")
#print(y)
nn.backprop(df, y, 0.00000001)
nn.SaveToFile()
nn.PrintWeights()