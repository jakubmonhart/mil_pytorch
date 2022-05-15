
import pandas as pd
import numpy as np
from sklearn import preprocessing


def get_raw_train_test_data(data_dir, test_index_dir):

    data = pd.read_excel(data_dir)
    test_indices  = np.loadtxt(test_index_dir, dtype=int)
    train_data = data[~data.bagID.isin(test_indices)]
    test_data = data[data.bagID.isin(test_indices)]

    x_train = train_data[train_data.columns[3:]].values
    normalizer = preprocessing.StandardScaler().fit(x_train)
    
    train_data[train_data.columns[3:]] = normalizer.transform(train_data[train_data.columns[3:]])
    test_data[test_data.columns[3:]] = normalizer.transform(test_data[test_data.columns[3:]])

    return train_data, test_data