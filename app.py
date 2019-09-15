import flask
import pickle

# Use pickle to load in the pre-trained model.
with open(f'model/weights.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/')
def main():
    return(flask.render_template('main.html'))

if __name__ == '__main__':
    app.run()