from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


def get_report(risk_level,
               model,
               data_prep_pipeline,
               input_vector,
               feature_importances,
               safe_limits,
               pred_probab):

    feature_name_dict = {
        'sbp': 'Blood pressure',
        'ldl': 'Cholestrole',
        'obesity': 'BMI',
        'adiposity': 'Adiposity index',
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
                new_value = input_vector[feature][0] - safe_limits[feature]


        elif feature == 'tobacco':
            age = input_vector['age'][0]
            excess_tobacco = input_vector[feature][0] - ((age - 10) * safe_limits[feature])
            if excess_tobacco > 0:
                impact = feature_importances[feature] * excess_tobacco
                new_value = (input_vector[feature][0] / (age - 10)) * (age - 5) # projected usage in next 5 years

        if max_impact < impact:
            max_impact = impact
            max_impact_feature = feature

        if max_impact == 0:

            if risk_level == 2:
                return 'Possibility of CVD'

            else:
                return 'Critical'

        else:

            input_vector[max_impact_feature] = new_value
            input_vector_prep = data_prep_pipeline.transform(input_vector)
            new_probab = model.predict_proba(input_vector_prep)[0][1]
            perc_improvement = round(((pred_probab - new_probab) / pred_probab), 3) * 100

            if max_impact_feature in ['sbp', 'ldl', 'obesity', 'adiposity']:
                report = (f'Most critical factor is {feature_name_dict[max_impact_feature]}. If you bring it down to the doctor \
                           prescribed limit of {safe_limits[max_impact_feature]}, risk decreases by \
                           {perc_improvement} %')

            elif max_impact_feature == 'alcohol':
                report = (f'Most critical factor is {feature_name_dict[max_impact_feature]}. If you bring it down to the doctor \
                           prescribed limit of {safe_limits[max_impact_feature]} units, risk decreases by \
                           {perc_improvement} %')

            elif max_impact_feature == 'tobacco':
                report = (f"Most critical factor is {feature_name_dict[max_impact_feature]}. If you don't tone down your tobacco consumption, \
                            in the next 5 years, your risk increases by {perc_improvement} %")

            return report