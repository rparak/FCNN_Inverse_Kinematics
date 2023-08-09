import sklearn.preprocessing

data = [1, 0, 1, 0, 1, 0]
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=range)
data_tmp = scaler.fit_transform(data)