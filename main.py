import numpy as np
from sklearn import preprocessing
data = np.array([[1, 2], [3, 4]]) 

#return 1 if greater than treshold 0 otherwise
binaryData= preprocessing.binarize(data, treshold = np.mean(data))

#return the normal distribution of the data
transformer = preprocessing.StandardScaler().fit(data)
transformer.transform(data)

#return according to the quantile range ( median = 0)
transformer = preprocessing.RobustScaler().fit(data)
transformer.transform(data)

#return (value - min) / (max -min)
scaler = MinMaxScaler().fit(data)
scaler.transform(data)
