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
		party_size = request.form['party_size']
		dist_ride = request.form['player_dist_ride']
		dist_walk = request.form['player_dist_walk']
		hitpoint = request.form['player_dmg']
		parameters = np.array([party_size,dist_ride,dist_walk,hitpoint]).reshape(1,-1)
		prediction = top10.predict(parameters)
		if (prediction == -1):
			return render_template('index.html', label = "You're an amateur, not even in top 10! Play more, improve your game.")
		else :
			prediction = top3.predict(parameters)
			if(prediction == 1):
				label = "You'll be amongst the top 3. You're an elite."
			else:
				if(top5.predict(parameters) == 1):
					label = "You'll be amongst the top 5. Good Game."
				else:
					label = "You'll be amongst the top 10. Try Harder."
			return render_template('index.html', label = label)
	
if __name__ == '__main__':
	# load ml models
	top10 = joblib.load('top10.pkl')
	top5 = joblib.load('top5.pkl')
	top3 = joblib.load('top3.pkl')
	# start api
	app.run(port=8000, debug=True)
