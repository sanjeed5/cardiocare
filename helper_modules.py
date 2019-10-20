import math
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


def get_report(model,
               data_prep_pipeline,
               input_vector,
               feature_importances,
               safe_limits,
               pred_probab):

    feature_name_dict = {
        'sbp': 'Blood pressure',
        'ldl': 'Cholestrole',
        'obesity': 'Obesity',
        'adiposity': 'Adiposity',
        'alcohol': 'Alcohol comsumption',
        'tobacco': 'Smoking'
    }

    impact = 0
    max_impact = 0
    max_impact_feature = None
    for feature in feature_importances:

        if feature in ['sbp', 'ldl', 'obesity', 'adiposity', 'alcohol']:
            if input_vector[feature][0] > safe_limits[feature]:
                impact = feature_importances[feature] * (input_vector[feature][0] - safe_limits[feature])

        elif feature == 'tobacco':
            age = input_vector['age'][0]
            excess_tobacco = input_vector[feature][0] - ((age - 10) * safe_limits[feature])
            if excess_tobacco > 0:
                impact = feature_importances[feature] * excess_tobacco

        if max_impact < impact:
            max_impact = impact
            max_impact_feature = feature

    if max_impact == 0:
        return 'You are at risk of CVD'

    else:
        report = f'You are at risk of CVD. Most critical factor is {feature_name_dict[max_impact_feature]}. '
        return report