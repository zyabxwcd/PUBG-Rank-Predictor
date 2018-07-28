# Importing the classifier
from sklearn.ensemble import RandomForestClassifier

# For downsampling
from sklearn.utils import resample

# For saving the model to disk
#from sklearn.externals import joblib

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Reading data 
data = pd.read_csv("C:\\agg_match_stats_0.csv", nrows=1000000)

# Extracting needed data
value_list = list(range(1,11))
df1 = data[data.team_placement.isin(value_list)].copy()
df2 = data[~data.team_placement.isin(value_list)].copy()
df1['team_placement'] = 1
df2['team_placement'] = -1
frames = [df1,df2]
df = pd.concat(frames)
df.drop(['date','match_id','match_mode','player_dist_ride','player_assists','player_name','team_id'], axis = 1, inplace = True)

# Checking for missing values
print(df.isnull().values.any())

# Checking for imbalances in cases for each class
print(df.team_placement.value_counts())

# Resampling imbalanced data
df_majority = df[df.team_placement==-1]
df_minority = df[df.team_placement!=-1]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=270120, # to match minority class
                                 random_state=42)  # reproducible results

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Randomly marking 70% rows for training
df_downsampled['is_train'] = np.random.uniform(0, 1, len(df_downsampled)) <= .70

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df_downsampled[df_downsampled['is_train']==True], df_downsampled[df_downsampled['is_train']==False]

features = train[train.columns[:7]]
y = train['team_placement']
x_test = test[test.columns[:7]]
y_test = test['team_placement']

# Training the model
clf = RandomForestClassifier(n_jobs=-1, n_estimators = 100, random_state=42, max_features=0.5)
clf.fit(features, y)

# Saving the model to disk
filename = 'top10_Prob.pkl'
joblib.dump(clf, filename)

preds = clf.predict(x_test)
print(pd.crosstab(test['team_placement'], preds, rownames=['Actual'], colnames=['Predicted']))
print ('RF accuracy: TRAINING', clf.score(features,y))
print ('RF accuracy: TESTING', clf.score(x_test,y_test))

# For additional accuracy scores
print (metrics.classification_report(test['team_placement'], predicted))