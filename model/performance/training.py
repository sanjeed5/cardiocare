import pickle
import numpy as np
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier as RF_normal
from daal4py.sklearn.ensemble import RandomForestClassifier as RF_intel


def get_training_time():

    df = pd.read_csv(r'C:\Users\user\cardiocare\model\performance\cardio.csv')
    df.columns = (['bp', 'tobaco', 'cholestrol', 'adiposity', 'fam_hist',
                   'type_a_beh', 'obesity', 'alcohol', 'age', 'Class'])

    x = df.drop('Class', axis = 1)
    y = df['Class']

    y = y.to_numpy()
    y = np.array([1 if i == 1 else 0 for i in y])

    clf_normal = RF_normal(n_estimators = 100, max_depth = 6, random_state = 0)
    clf_intel = RF_intel(n_estimators = 100, max_depth = 6, random_state = 0)

    start_intel = time()
    clf_intel.fit(x, y)
    end_intel = time()

    start_normal = time()
    clf_normal.fit(x, y)
    end_normal = time()

    print(f'Time taken (intel) : {end_intel - start_intel}')
    print(f'Time taken (normal): {end_normal - start_normal}')


if __name__ == '__main__':

    get_training_time()

