import pandas as pd
import pandas_profiling as pp
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from daal4py.sklearn.ensemble import RandomForestClassifier


def accuracy(y_true, y_pred):

    N = y_true.shape[0]
    y_true_copy = y_true.copy()
    if(type(y_true) == pd.Series):
        y_true_copy = y_true_copy.to_numpy()

    numerator = 0
    for i in range(N):
        if(y_true_copy[i] == y_pred[i]):
            numerator += 1

    return numerator / N


def training(csv_file, final_model_file_name):

    df = pd.read_csv(csv_file)
    df.columns = (['bp', 'tobaco', 'cholestrol', 'adiposity', 'fam_hist',
                   'type_a_beh', 'obesity', 'alcohol', 'age', 'Class'])

    x = df.drop('Class', axis = 1)
    y = df['Class']

    y = y.to_numpy()
    y = np.array([1 if i == 1 else 0 for i in y])

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                            test_size = 0.2, random_state = 42)

    scaler = MinMaxScaler()
    scaler.fit_transform(x_train)
    scaler.transform(x_test)

    clf = RandomForestClassifier(n_estimators = 100, max_depth = 6, random_state = 0)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    results_dict = dict()
    results_dict['train_accuracy'] = accuracy(y_train, y_train_pred)
    results_dict['test_accuracy'] = accuracy(y_test, y_test_pred)

    with open(final_model_file_name, 'wb') as file:
        pickle.dump(clf, file)

    return results_dict



