# Importing classifier
from sklearn.ensemble import RandomForestClassifier

# For saving the model to disk
from sklearn.externals import joblib

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Reading data from csv
data = pd.read_csv("C:\\agg_match_stats_0.csv", nrows=1000000)

# Extracting needed data
value_list = list(range(1,11))
df = data[data.team_placement.isin(value_list)].copy()
df.drop(['date','match_id','match_mode','player_dist_ride','player_assists','player_name','team_id'], axis = 1, inplace = True)

# Selecting features and target
features = df[df.columns[:7]]
target = df['team_placement']

# Training the classifier
clf = RandomForestClassifier(n_jobs=-1, n_estimators = 100, random_state=42, max_features=0.5)
clf.fit(features, target)

# Saving the model to disk
filename = 'top10_ProbPred.pkl'
joblib.dump(clf, filename)
