from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


def get_report(risk_level, model, data_prep_pipeline, input_vector, feature_importances, safe_limits):

    max_impact = 0
    max_impact_feature = None
    for feature in feature_importances:

        if feature in ['sbp', 'ldl', 'obesity', 'adiposity', 'alcohol']:
            if input_vector[feature][0] > safe_limits[feature]:
                impact = feature_importances[feature] * (input_vector[feature][0] - safe_limits[feature])
                new_value = input_vector[feature][0] - safe_limits[feature]


        elif feature == 'tobacco':
            excess_cigs = input_vector[feature][0] * 20 - input_vector['age'][0] * safe_limits[feature]
            if excess_cigs > 0:
                impact = feature_importances[feature] * excess_cigs
                new_value = input_vector[feature][0] - safe_limits[feature]


        if max_impact < impact:
            max_impact = impact
            max_impact_feature = feature

        if max_impact == 0:
            return None

        else:
            return max_impact_feature,

            # input_vector[max_impact_feature] = new_value
            # input_vector_prep = data_prep_pipeline.transform(input_vector)
            # new_probab = model.predict_proba(input_vector_prep)[0][1]
            # perc_improvement = round((pred_probab - new_probab) / pred_probab, 2)