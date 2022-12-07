from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from pandas import read_csv
import pandas as pd
import numpy as np

__all__ = ['smote_func']


def smote_func(raw_data_file_path):
    """
    This function is used to deal with the imbalanced data set of
    the multivariate classification problem, And make each category
    have the same number of examples, that is, the number of examples
    in the largest category.

    Parameters
    ----------

    raw_data_file_path : str.
        Filename of a .csv file containing features (geographic location)
        and label data(i.e. risk level).

    Returns
    -------

    pandas.DataFrame
        DataFrame consists of 'easting', 'northing' and flood risk
        classification labels, this is a new balanced dataset.
    """

    # define the data set location
    url = raw_data_file_path
    # load the csv file as a data frame
    dataset = read_csv(url, header=0)
    dataset = pd.DataFrame(pd.DataFrame(dataset)[['easting', 'northing',
                                                  'riskLabel']])
    data = dataset.values
    # split into input and output elements
    xdata = data[:, :-1]
    y_data = data[:, -1]
    # label encode the target variable
    labelencoder = LabelEncoder()
    labelencoder.fit(y_data)
    y_data = labelencoder.transform(y_data)
    # y = LabelEncoder().fit_transform(y)
    # transform the dataset
    oversample = SMOTE()
    xdata, y_data = oversample.fit_resample(xdata, y_data)
    # summarize distribution
    counter = Counter(y_data)
    for elem, val in counter.items():  # noqa
        per = val / len(y_data) * 100  # noqa
        # print(f'Class={elem}, n={val} ({per})')

    xdata = pd.DataFrame(xdata)
    y_data = labelencoder.inverse_transform(y_data)
    y_data = pd.DataFrame(y_data)

    # concatenating X and y along columns
    horizontal_concat = pd.concat([xdata, y_data], axis=1)
    data_after_smote = pd.DataFrame(np.array(horizontal_concat),
                                    columns=('easting', 'northing',
                                             'riskLabel'))
    data_after_smote['riskLabel'] = data_after_smote['riskLabel'].astype(int)

    return data_after_smote
