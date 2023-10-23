# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Reading the dataset
heart = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Preparing the features and target
X = heart.drop('DEATH_EVENT', axis=1)
Y = heart['DEATH_EVENT']

# Initializing the Random Forest Classifier
clf = RandomForestClassifier()

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Training the classifier
clf.fit(X_train, Y_train)

# Predicting on the test set
predicted_y = clf.predict(X_test)
print(f'predicted y value : {predicted_y}')

# Checking the accuracy on the training and testing sets
clf.score(X_train, Y_train)
clf.score(X_test, Y_test)

# Evaluating the model with different numbers of estimators
for _ in range(10, 200, 10):
    print(f'Running Model with {_} estimators')
    clf = RandomForestClassifier(n_estimators=_).fit(X_train, Y_train)
    print(f'Accuracy is : {clf.score(X_test, Y_test)}')

# Saving the trained model using Pickle
pickle.dump(clf, open('Heart Failure Predictor.pkl', 'wb'))

# Loading the saved model
load_model = pickle.load(open('Heart Failure Predictor.pkl', 'rb'))
print(f'Loaded Model Score: {load_model.score(X_test, Y_test)}')
