#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")
from time import time

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest

## Definitions
def plot_scatter(data,x_label,y_label):
    ptr1 = [data[i][0] for i in range(len(data))]
    ptr2 = [data[i][1] for i in range(len(data))]
    ptr3 = [data[i][2] for i in range(len(data))]
    ptr4 = [data[i][3] for i in range(len(data))]
   # ptr5 = [data[i][4] for i in range(len(data))]
    #ptr6 = [data[i][5] for i in range(len(data))]
    matplotlib.pyplot.scatter(ptr1,ptr2,color='r',label='total_stock_value')
    matplotlib.pyplot.scatter(ptr1,ptr3,color='y',label='exercised_stock_options')
    matplotlib.pyplot.scatter(ptr1,ptr4,color='b',label='bonux')
   # matplotlib.pyplot.scatter(ptr1,ptr5,color='green',label='long_term_incentive')
   # matplotlib.pyplot.scatter(ptr1,ptr6,color='orange',label='from_poi_to_this_person')


    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.show()


def outlier_rem():
    rerun = True
    while(rerun):
        rerun = False
        outliers = []
        for key in data_dict:
            if data_dict[key]['salary'] > 2.5*pow(10,7) and data_dict[key]['salary'] != 'NaN':
                outliers += [key]
                rerun = True
        if rerun == True:
            for item in outliers:
                print('removing key-value pair %s' % item)
                del data_dict[item]


def feature_scale(features):
    shape = [len(features),len(features[0])]
    for j in range(shape[1]):
        tmp_list = [features[i][j] for i in range(shape[0])]
        min_max = max(tmp_list) - min(tmp_list)
        for i in range(shape[0]):
            features[i][j] = float(tmp_list[i]-min(tmp_list))/min_max
    return features

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_stock_value','exercised_stock_options','bonus','long_term_incentive'] # You will need to use more features
features_list = features_list[:]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## Refer list of POIs
"""
fp = open("poi_names.txt","r")
a = fp.readlines()
for i in range(len(a)):
    a[i] = a[i].replace(" ", '')
tmp =''.join(a[2:])
print (len(tmp.split())) ## Number of POI that should be in the list
print ("Nunber of datapoints %d" % len(data_dict.keys()))
"""
sum_poi = 0
for key in data_dict:
    if data_dict[key]["poi"] == 1:
        sum_poi += 1
print ("Number of POI in the dataset %d" % sum_poi)
#print (data_dict[data_dict.keys()[0]])

### Task 2: Remove outliers
## old plot

data_tmp = featureFormat(data_dict, features_list, sort_keys = True, remove_NaN=True)
labels, features = targetFeatureSplit(data_tmp)
#plot_scatter(features,'salary','Y')

## Remvoing
outlier_rem()

## new plot without outliers

data_tmp = featureFormat(data_dict, features_list, sort_keys = True, remove_NaN=True)
labels, features = targetFeatureSplit(data_tmp)
#plot_scatter(features,'salary','Y')

## Scaling of data and visualizing it

features_s = feature_scale(features)
#plot_scatter(features,'salary','Y')


## Testing a simple classifier

## Without scaling
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
#clf_old = GaussianNB()
clf_old = MultinomialNB()
clf_old.fit(features_train,labels_train)
pred_old = clf_old.predict(features_test)
score_old = accuracy_score(labels_test,pred_old)
print("Accuracy without scaling of features %0.3f" % score_old)


## With scaling
features_train, features_test, labels_train, labels_test = train_test_split(features_s, labels, test_size=0.3, random_state=42)
#clf_new = GaussianNB()
clf_new = SVC()
clf_new.fit(features_train,labels_train)
pred_new = clf_new.predict(features_test)
score_new = accuracy_score(labels_test,pred_new)
print("Accuracy with scaling of features %0.3f" % score_new)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
my_dataset = data_dict ## Can be used later for rescaling

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

nPC = len(features_list)
features_train_pca = []
features_test_pca = []
t0 = time()

### Using Pipeline with GridCVSearch to find the best esimator

### SVC

selector = SelectKBest(k=2)

pca = PCA()


combinedFeatures = FeatureUnion([("pca", pca), ("univ_select", selector)])

#tmp_features_train = combinedFeatures.fit(features_train,labels_train).transform(features_train)
#tmp_features_test = combinedFeatures.transform(features_test)

#estimators = [('reduce_dim',combinedFeatures),('clf1',SVC())]
"""
print ("------------------NEW CLASSIFIER---SVC------------------")

estimators = [('reduce_dim',PCA()),('clf1',SVC())]
pipe = Pipeline(estimators)
t0 = time()
params = dict(reduce_dim__n_components=[2,3,4,5],
              #reduce_dim__univ_select__k=[1,2,3,4,5],
              clf1__C = [1e3, 5e3, 1e4, 5e4, 1e5],
              clf1__gamma=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              clf1__kernel=['linear','rbf','sigmoid','poly'],
              clf1__class_weight=['balanced'])

clf1 = GridSearchCV(pipe,param_grid = params,cv=3,return_train_score=True,n_jobs=-1)
clf1 = clf1.fit(features_train,labels_train)
pred1 = clf1.predict(features_test)
print ("Time taken %0.3f s" % round(time()-t0,3))
print('Accuracy score: ', format(accuracy_score(labels_test, pred1)))
print('Precision score: ', format(precision_score(labels_test, pred1)))
print('Recall score: ', format(recall_score(labels_test, pred1)))
print('F1 score: ', format(f1_score(labels_test, pred1)))
#print (clf1.best_estimator_)

print ("------------------NEW CLASSIFIER---Gaussian------------------")

### GaussianNB
t0 = time()
estimators = [('reduce_dim',PCA()),('clf2',DecisionTreeClassifier())]
#estimators = [('reduce_dim',combinedFeatures),('clf2',GaussianNB())]
pipe = Pipeline(estimators)
params = dict(reduce_dim__n_components=[2,3,4,5],
              #reduce_dim__univ_select__k=[1,2,3,4,5],
              reduce_dim__random_state=[42])

clf2 = GridSearchCV(pipe,param_grid = params,cv=3,return_train_score=True,n_jobs=-1)
clf2 = clf2.fit(features_train,labels_train)
pred2 = clf2.predict(features_test)
print ("Time taken %0.3f s" % round(time()-t0,3))
print('Accuracy score: ', format(accuracy_score(labels_test, pred2)))
print('Precision score: ', format(precision_score(labels_test, pred2)))
print('Recall score: ', format(recall_score(labels_test, pred2)))
print('F1 score: ', format(f1_score(labels_test, pred2)))
#print (clf2.best_estimator_)
"""
print ("------------------NEW CLASSIFIER---DT------------------")

### Decision Trees
t0 = time()
estimators = [('reduce_dim',PCA()),('clf3',DecisionTreeClassifier())]
#estimators = [('reduce_dim',combinedFeatures),('clf3',DecisionTreeClassifier())]
pipe = Pipeline(estimators)

params = dict(reduce_dim__n_components=[1,2,3],
              #reduce_dim__univ_select__k=[1,2,3,4,5],
              reduce_dim__random_state=[42],
              clf3__criterion=['gini','entropy'],
              clf3__min_samples_split=[x for x in range(2,16)])

clf3 = GridSearchCV(pipe,param_grid = params,cv=4,return_train_score=True,n_jobs=-1)
clf3 = clf3.fit(features_train,labels_train)
pred3 = clf3.predict(features_test)
print ("Time taken %0.3f s" % round(time()-t0,3))
print('Accuracy score: ', format(accuracy_score(labels_test, pred3)))
print('Precision score: ', format(precision_score(labels_test, pred3)))
print('Recall score: ', format(recall_score(labels_test, pred3)))
print('F1 score: ', format(f1_score(labels_test, pred3)))
#print (clf3.best_estimator_
"""
print ("------------------NEW CLASSIFIER---ADABOOST------------------")

## AdaBoost
t0 = time()
#estimators = [('reduce_dim',combinedFeatures),('clf4',AdaBoostClassifier())]
estimators = [('reduce_dim',PCA()),('clf4',AdaBoostClassifier())]
pipe = Pipeline(estimators)

params = dict(reduce_dim__n_components=[2,3,4,5],
              #reduce_dim__univ_select__k=[1,2,3,4,5],
              reduce_dim__random_state=[42],
              clf4__algorithm=['SAMME', 'SAMME.R'],
              clf4__random_state=[42],
              clf4__n_estimators=[x for x in range(20,200,10)])

clf4 = GridSearchCV(pipe,param_grid = params,cv=3,return_train_score=True,n_jobs=-1)
clf4 = clf4.fit(features_train,labels_train)
pred4 = clf4.predict(features_test)
print ("Time taken %0.3f s" % round(time()-t0,3))
print('Accuracy score: ', format(accuracy_score(labels_test, pred4)))
print('Precision score: ', format(precision_score(labels_test, pred4)))
print('Recall score: ', format(recall_score(labels_test, pred4)))
print('F1 score: ', format(f1_score(labels_test, pred4)))
#print (clf4.best_estimator_)

print ("------------------NEW CLASSIFIER---RandomForest------------------")

### Random Forest
t0 = time()
#estimators = [('reduce_dim',combinedFeatures),('clf5',RandomForestClassifier())]
estimators = [('reduce_dim',PCA()),('clf5',RandomForestClassifier())]
pipe = Pipeline(estimators)

params = dict(reduce_dim__n_components=[2,3,4,5],
              #reduce_dim__univ_select__k=[1,2,3,4,5],
              reduce_dim__random_state=[42],
              clf5__n_estimators=[x for x in range(2,11)],
              clf5__random_state=[42],
              clf5__criterion=['gini','entropy'],
              clf5__min_samples_split=[x for x in range(2,8)])

clf5 = GridSearchCV(pipe,param_grid = params,cv=3,return_train_score=True,n_jobs=-1)
clf5 = clf5.fit(features_train,labels_train)
pred5 = clf5.predict(features_test)
print ("Time taken %0.3f s" % round(time()-t0,3))
print('Accuracy score: ', format(accuracy_score(labels_test, pred5)))
print('Precision score: ', format(precision_score(labels_test, pred5)))
print('Recall score: ', format(recall_score(labels_test, pred5)))
print('F1 score: ', format(f1_score(labels_test, pred5)))
#print (clf5.best_estimator_)
"""

"""
print ("------------------NEW CLASSIFIER---MultinomialNB------------------")

### MultinomialNB
t0 = time()
#estimators = [('reduce_dim',combinedFeatures),('clf5',RandomForestClassifier())]
estimators = [('reduce_dim',PCA()),('clf6',MultinomialNB())]
pipe = Pipeline(estimators)

params = dict(reduce_dim__n_components=[2,3,4,5],
              #reduce_dim__univ_select__k=[1,2,3,4,5],
              reduce_dim__random_state=[42])
              #clf6__n_estimators=[x for x in range(2,11)],
              #clf6__random_state=[42],
              #clf6__criterion=['gini','entropy'],
              #clf6__min_samples_split=[x for x in range(2,8)])

clf6 = GridSearchCV(pipe,param_grid = params,cv=3,return_train_score=True,n_jobs=1)
clf6 = clf6.fit(features_train,labels_train)
pred6 = clf6.predict(features_test)
print ("Time taken %0.3f s" % round(time()-t0,3))
print('Accuracy score: ', format(accuracy_score(labels_test, pred6)))
print('Precision score: ', format(precision_score(labels_test, pred6)))
print('Recall score: ', format(recall_score(labels_test, pred6)))
print('F1 score: ', format(f1_score(labels_test, pred6)))
print (clf6.best_estimator_)
"""
"""
for i in range(1,nPC):
    print "Extracting the top %d eigenfaces from %d faces" % (i, nPC)
    pca = PCA(n_components=i, whiten=True,random_state=41+i).fit(features_train)
    print "done in %0.3fs" % (time() - t0)
    eigenRatios = pca.explained_variance_ratio_
    for j in range(i):
        print ("Eigen Value for",j,eigenRatios[j])
    print "Projecting the input data on the eigenfaces orthonormal basis"
    t0 = time()
    features_train_pca+=[pca.transform(features_train)]
    features_test_pca+=[pca.transform(features_test)]
    print "done in %0.3fs" % (time() - t0)
"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print("--------------------------------------")
clf = clf3
print(clf.best_estimator_)
"""
print ("------------------FINAL CLASSIFIER------------------")

pca = PCA(n_components = 2,random_state = 42)
features_train = pca.fit(features_train).transform(features_train)
features_test = pca.transform(features_test)

#clf = SVC(C=1000.0,kernel='poly',gamma=0.1)
clf = DecisionTreeClassifier(min_samples_split=15)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print ("Time taken %0.3f s" % round(time()-t0,3))
print('Accuracy score: ', format(accuracy_score(labels_test, pred)))
print('Precision score: ', format(precision_score(labels_test, pred)))
print('Recall score: ', format(recall_score(labels_test, pred)))
print('F1 score: ', format(f1_score(labels_test, pred)))
print (clf)
"""
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

