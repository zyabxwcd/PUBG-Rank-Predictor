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
		knockdowns = request.form['player_dbno']
		dist_walk = request.form['player_dist_walk']
		hitpoint = request.form['player_dmg']
		player_kills = request.form['player_kills']
		survival_time = request.form['player_survival_time']
		parameters = np.array([game_size,party_size,knockdowns,dist_walk,hitpoint,player_kills,survival_time])
		prediction = model1.predict(parameters.reshape(1, -1))
		if (prediction == -1):
			return render_template('index.html', label = "You're an amateur! Play more to be amongst the top 10.")
		else :
			prediction = model2.predict_proba(parameters.reshape(1,-1))
			label = "You're an elite amongst the top 10. Rank Probability :" + str(prediction)
			return render_template('index.html', label = label)
	
if __name__ == '__main__':
	# load ml model
	model1 = joblib.load('top10_Pred.pkl')
	model2 = joblib.load('top10_ProbPred.pkl')
	# start api
	app.run(port=8000, debug=True)
