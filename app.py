import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model.
with open(f'model/weights.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        bp = flask.request.form['bp']
        tobaco = flask.request.form['tobaco']
        cholestrol = flask.request.form['cholestrol']
        adiposity = flask.request.form['adiposity']
        fam_hist = flask.request.form['fam_hist']
        type_a_beh = flask.request.form['type_a_beh']
        obesity = flask.request.form['obesity']
        alcohol = flask.request.form['alcohol']
        age = flask.request.form['age'] 
        
        input_variables = pd.DataFrame([[bp, tobaco, cholestrol, adiposity, fam_hist, type_a_beh, obesity, alcohol, age]], 
                                        columns=['bp', 'tobaco', 'cholestrol', 'adiposity', 'fam_hist', 'type_a_beh', 'obesity', 'alcohol', 'age'], 
                                        dtype=float)

        prediction = model.predict(input_variables)[0]

        return flask.render_template('main.html',
                                     original_input={'Bp':bp,
                                                    'Tobaco':tobaco,
                                                    'Cholestrol':cholestrol,
                                                    'Adiposity':adiposity,
                                                    'Fam_hist':fam_hist,
                                                    'Type_a_beh':type_a_beh,
                                                    'Obesity':obesity,
                                                    'Alcohol':alcohol,
                                                    'Age':age},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()