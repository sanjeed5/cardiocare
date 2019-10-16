import flask
import pickle
import pandas as pd
import numpy as np

# Use pickle to load pre-trained model, data preparation pipeline, probability threshold
with open(f'model/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'model/data_prep_pipeline.pkl', 'rb') as g:
    data_prep_pipeline = pickle.load(g)

with open(f'model/normal_threshold.pkl', 'rb') as h:
    normal_threshold = pickle.load(h)

with open(f'model/critical_threshold.pkl', 'rb') as i:
    critical_threshold = pickle.load(i)

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        bp = flask.request.form['bp']
        tobacco = flask.request.form['tobacco']
        cholestrol = flask.request.form['cholestrol']
        adiposity = flask.request.form['adiposity']
        fam_hist = flask.request.form['fam_hist']
        type_a_beh = flask.request.form['type_a_beh']
        obesity = flask.request.form['obesity']
        alcohol = flask.request.form['alcohol']
        age = flask.request.form['age']

        input_vector = pd.DataFrame([[bp, tobacco, cholestrol, adiposity, fam_hist, type_a_beh, obesity, alcohol, age]],
                       columns = ['sbp', 'tobacco', 'ldl',	'adiposity', 'famhist',	'typea', 'obesity',	'alcohol', 'age'], dtype=float)

        input_vector = data_prep_pipeline.transform(input_vector)
        pred_probab = model.predict_proba(input_vector)[0][1]

        if pred_probab < normal_threshold:
            result = 'Normal'

        elif pred_probab <= critical_threshold and pred_probab >= normal_threshold:
            result = 'Possibility of heart disease'

        else:
            result = 'Critical'

        return flask.render_template('main.html',
                                     original_input={'Bp':bp,
                                                     'Tobacco':tobacco,
                                                     'Cholestrol':cholestrol,
                                                     'Adiposity':adiposity,
                                                     'Fam_hist':fam_hist,
                                                     'Type_a_beh':type_a_beh,
                                                     'Obesity':obesity,
                                                     'Alcohol':alcohol,
                                                     'Age':age},
                                     result=result,
                                     )

if __name__ == '__main__':
    app.run()