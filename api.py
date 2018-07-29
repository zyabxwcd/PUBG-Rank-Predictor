import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
		game_size = request.form['game_size']
		party_size = request.form['party_size']
		player_assists = request.form['player_assists']
		knockdowns = request.form['player_dbno']
		dist_walk = request.form['player_dist_walk']
		hitpoint = request.form['player_dmg']
		player_kills = request.form['player_kills']
		survival_time = request.form['player_survival_time']
		parameters = np.array([game_size,party_size,player_assists,knockdowns,dist_walk,hitpoint,player_kills,survival_time])
		prediction = model.predict(parameters.reshape(1, -1))
		return render_template('index.html', label = prediction)
	
if __name__ == '__main__':
	# load ml model
	model = joblib.load('finalized_model.pkl')
	# start api
	app.run(port=8000, debug=True)
