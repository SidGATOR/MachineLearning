import numpy as np
import pandas as pd
from time import time
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,fbeta_score,make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
# Import functionality for cloning a model
from sklearn.base import clone

import visuals as vs
"""
# Pretty display for notebooks
%matplotlib inline
"""
### Definition ###
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time()
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time()

    # Calculate the training time
    results['train_time'] = round(end-start,3)

    # Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # Calculate the total prediction time
    results['pred_time'] = round(end-start,3)

    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)

    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test,predictions_test)

    # Compute F-score on the the first 300 training samples using fbeta_score()
    beta = 0.5
    results['f_train'] = fbeta_score(y_train[:300],predictions_train,beta)

    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test,predictions_test,beta)

    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    # Return the results
    return results

### Loading the census data ###
data = pd.read_csv("census.csv")

print data.head()


### Basic infomation regarding the dataset ###
n_records = data.shape[0]

n_greater_50k = data[(data['income'] == '>50K')].shape[0]

n_atmost_50k = data[(data['income'] == '<=50K')].shape[0]

percentage = float(n_greater_50k)/n_records * 100

print "Total number of Records: {}" .format(n_records)
print "Number of people earning more that 50K: {}" .format(n_greater_50k)
print "Number of people earning at most 50K: {}" .format(n_atmost_50k)
print "Percentage of people earing more than 50K: {: .2f}%".format(percentage)

### Spliting the data in features and labels ###
raw_label = data['income']
raw_features = data.drop('income', axis=1)

# Visualize skewed continuous features of original data
vs.distribution(data)

### Since the data fields of capital loss and capital gain are either very low or high, we need to do a logarithming tranformation ###
### This way, the data will not undermine the classification model ###
skewed = ['capital-gain','capital-loss']
features_log_transformed = pd.DataFrame(data=raw_features)
features_log_transformed[skewed] = raw_features[skewed].apply(lambda x: np.log(x+1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)

### Feature Scaling ###
numerical_list = ['age','education-num','capital-gain','capital-loss','hours-per-week']
### Initializing scaler ###
scaler = MinMaxScaler()
features_log_transformed_s = pd.DataFrame(data=features_log_transformed)
features_log_transformed_s[numerical_list] = scaler.fit_transform(features_log_transformed[numerical_list])

### Converting catagorical datatype to numerical
final_features = pd.get_dummies(data=features_log_transformed_s)

income = pd.get_dummies(data=raw_label)

print "Columns before one-hot-encoding: {}" .format(len(list(features_log_transformed_s.columns)))
print "Columns after one-hot-encoding: {}" .format(len(list(final_features.columns)))

### Spliting Training and Testing data ###
features_train, features_test, labels_train,labels_test = train_test_split(final_features,income,test_size=0.2,random_state=42)

print "Training Set has: {} datapoints" .format(features_train.shape[0])
print "Testing Set has: {} datapoints" .format(features_test.shape[0])

### In order to test the performance of our classifier, we need to make a chance based model which can select a datapoin at random and classifiy it based on some hypothesis
### These is a need for such a classifier as this will allow us to keep a track of progress made other classifiers which will do better than just chance
### In this example, there are more number of people that are below the 50K mark. So, if we keep a hypothesis that every one earns more than 50K, we can find the baseline
### performance of a chance based classifier
chance_pred = np.ones(income.shape[0])
tmp_labels = np.array(income[">50K"])
accuracy = accuracy_score(tmp_labels,chance_pred) * 100
f1Score = f1_score(tmp_labels,chance_pred)
fBeta = fbeta_score(tmp_labels,chance_pred,0.5)
print "Accuracy of predictor: {:,.2f}%" .format(accuracy)
print "F1-Score: {:,.3f}" .format(f1Score)
print "F-Beta Score: {:,.4f}" .format(fBeta)


# Initialize the three models
clf_A = DecisionTreeClassifier()
clf_B = SVC()
clf_C = AdaBoostClassifier()

# Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100
# HINT: samples_1 is 1% of samples_100
samples_100 = len(labels_train)
samples_10 = len(labels_train[:int(0.1*samples_100)])
samples_1 = len(labels_train[:int(0.01*samples_100)])

### Converting DataFrame to List for sklearn
fearures_train = np.array(features_train)
features_test = np.array(features_test)
labels_train = np.array(labels_train['>50K'])
labels_test = np.array(labels_test['>50K'])

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, features_train, labels_train, features_test, labels_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fBeta)


# Initialize the classifier
#dtf = DecisionTreeClassifier()
clf = AdaBoostClassifier(random_state=42)

# Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'n_estimators' : [1000,1250,1500],
              'algorithm' : ['SAMME','SAMME.R']}

# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score,beta=0.5)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(estimator=clf,param_grid=parameters,scoring=scorer,n_jobs=-1)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(features_train,labels_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(features_train, labels_train)).predict(features_test)
best_predictions = best_clf.predict(features_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(labels_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(labels_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(labels_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(labels_test, best_predictions, beta = 0.5))

"""
### Manual Feature priority calculation ###
tmp_clf = AdaBoostClassifier()
tmp_clf = tmp_clf.fit(X_train,y_train)
tmpL = tmp_clf.feature_importances_
columnL = zip(tmpL,features_final.columns)
columnL = sorted(columnL)[::-1]
top_five = columnL[:5]
print top_five
"""

# Extract the feature importances using .feature_importances_
importances = best_clf.feature_importances_

X_train,X_test,y_train,y_test = train_test_split(final_features ,income,test_size=0.2,random_state=42)

# Plot
vs.feature_plot(importances, X_train, y_train)

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

### Converting the DataFrame to numpy for sklearn usage
X_train_reduced = np.array(X_train_reduced)
X_test_reduced = np.array(X_test_reduced)
y_train = np.array(y_train['>50K'])
y_test = np.array(y_test['>50K'])

# Train on the "best" model found from grid search earlier
start = time()
clf = (clone(best_clf)).fit(X_train_reduced, y_train)
end = time()
print "Time to Train: {:,.3f}s" .format(round(end-start,3))
print "----------"
# Make new predictions
start = time()
reduced_predictions = clf.predict(X_test_reduced)
end = time()
print "Time to Test: {:,.3f}s" .format(round(end-start,3))
print "----------"

# Report scores from the final model using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))
