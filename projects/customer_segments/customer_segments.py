import numpy as np
import pandas as pd
import visuals as vs
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

### Importing raw data from csv files ###
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Channel','Region'],axis=1,inplace=True)
    print "---------------------------"
    print "Whole customer dataset has {} samples with {} features each" .format(*data.shape)
    print "---------------------------"
except:
    raise ValueError("Dataset is missing...")


### Statistical Analysis of dataset ###
print "---------Statistical Data for analysis---------"
print data.describe()
print "---------------------------"

### Taking a small sample to analyze ###
print "---------Taking a small subset---------"
indices = [34,39,37]
sample = pd.DataFrame(data=data.loc[indices],columns=data.keys()).reset_index(drop=True)
print sample
print "---------------------------"
### Data Relevance ###
fresh = pd.DataFrame(data=data['Fresh'])
new_data = data.drop('Fresh',axis=1)

## Doing regression relevance check for every feature ##
labels = list(data.columns)
print "---------Feature Relevance Check---------"
for key in labels:
    label = pd.DataFrame(data=data[key])
    features = data.drop(key,axis=1)
    ## Spliting data into Train and Test sets

    X_train, X_test, y_train, y_test = train_test_split(features,label,test_size=0.25,random_state=42)

    ## Regressor ##
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(np.array(X_train),np.array(y_train))

    score = regressor.score(np.array(X_test),np.array(y_test))

    print "Score for {} is: {:,.3f}" .format(key,score)

## Plotting scatter-matrix for features ##
print "---------Scatter-matrix of features---------"
#scatter_matrix(data,alpha=0.3,figsize=(14,8),diagonal='kde');
#plt.show()
print "---------------------------"

### Data Preprocessing ###
print "---------Preprocessing-data---------"
print "---------------------------"
## Since the data shows non normal distribution, we can employ logarithm tranformation to make it normal as much as possible ##
## We have added 1 so that the value of datapoint with x = 0 is valid ##
log_data = data.apply(lambda x: np.log(1+x))

log_sample = sample.apply(lambda x: np.log(1+x))

## Scatter plot ##
print "---------Scatter-matrix of log_data---------"
#scatter_matrix(log_data,alpha=0.3,figsize=(14,8),diagonal='kde')
#plt.show()
print "---------------------------"

### Outlier Removal ###
outliers = []

for feature in log_data.keys():

    ## 25th Percentile
    Q1 = np.percentile(log_data[feature],25)

    ## 75th Percentile
    Q3 = np.percentile(log_data[feature],75)

    ## Inter Quartile step
    step = 1.5 * (Q3 - Q1)

    ##Outliers
    print "outlier for {} feature\n" .format(feature)
    print log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    print "---------------------------"

    outliers += list((log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]).index.values)

# Remove the outliers, if any were specified
outliers = sorted(list(set(outliers)))
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

print "---------scatter-matrix for good data---------"
#scatter_matrix(good_data,alpha=0.3,figsize=(14,8),diagonal='kde')
#plt.show()
print "---------------------------"

### Feature Transfomation ###
len_features = len(list(good_data.columns))
pca = PCA(n_components=len_features)

pca_samples = pca.fit(good_data).transform(log_sample)

## Generating a PCA result plot ##
pca_results = vs.pca_results(good_data,pca)
plt.show()

### Dimension Reduction ###
pca = PCA(n_components=2)

reduced_data = pca.fit_transform(good_data)

pca_samples = pca.transform(log_sample)

## Creating a reduced DataFrame
reduced_data = pd.DataFrame(data=reduced_data,columns=['Dimension 1','Dimension 2'])

## Biplot is ascatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case Dimension 1 and Dimension 2). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features
print "---------Biplot of 1,2 PCs with original features"
vs.biplot(good_data, reduced_data, pca)
plt.show()
print "---------------------------"

### Implementing Clusters ###
score_card = []
n_clusters = [x for x in range(2,17)]
for x in n_clusters:
    clusterer = KMeans(n_clusters=x,random_state=42)

    preds = clusterer.fit_predict(reduced_data)

    centers = clusterer.cluster_centers_

    sample_preds = clusterer.predict(pca_samples)

    ## Calculate the Silhouette Score for x cluster centers
    score = silhouette_score(reduced_data,preds)
    score_card += [(score,x)]
    print "Silhouette Score of {} clusters is: {:,.3f}" .format(x,score)

print "---------------------------"

score_card = sorted(score_card)[::-1]
print "---------------------------"
## Running for Optimal number of cluster with highest Silhouette Score ##
clusterer = KMeans(n_clusters=score_card[0][1],random_state=42)

preds = clusterer.fit_predict(reduced_data)

centers = clusterer.cluster_centers_

sample_preds = clusterer.predict(pca_samples)

score = silhouette_score(reduced_data,preds)

print "Silhouette Score of optimal number of centers: {:,.3f}" .format(score)
print "---------------------------"

## Cluster Visualization ##
print "---------Cluster Visualization of reduced_data---------"
vs.cluster_results(reduced_data, preds, centers, pca_samples)
plt.show()
print "---------------------------"

### Data Recovery ###

## Inverse transform the centers ##
log_centers = pca.inverse_transform(centers)

## Exponentiate the centers ##
true_centers = np.exp(log_centers-1)


## Display the true centers ##
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
print "---------Centers of the clusters---------"
print true_centers.head()
print "---------------------------"

