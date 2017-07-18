import numpy as np
import pandas as pd

### Definitions
def accuracy_score(true,pred):

    if len(true) != len(pred):
        return "Number of outcomes not equal to number of predictions made"

    else:
        return "Predictions have and accuracy of {:0.3f}%".format((true==pred).mean()*100)


### Prediction Model 0: No one survives
def prediction_0(data):

    prediction = []
    for _, passenger in data.iterrows():
        prediction += [0]

    return prediction

### Prediction Model 1: Female survive
def prediction_1(data):

    predictions = []

    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            predictions += [1]
        else:
            predictions += [0]

    return predictions

### Prediction Model 2: Males below 10 survive

def prediction_2(data):

    predictions = []

    for _, passenger in data.iterrows():

        if passenger['Sex'] == 'female':
            predictions += [1]

        else:
            if passenger['Age'] < 10 :
                predictions += [1]
            else:
                predictions += [0]

    return predictions

### Prediction Model 3: Accuracy more than 80%

def prediction_3(data):

    predictions = []

    for _, passenger in data.iterrows():

        if passenger['Sex'] == 'female':
            if passenger['Pclass'] <= 2:
                predictions += [1]
            else:
                if passenger['Age'] >= 60:
                    predictions += [1]
                else:
                    predictions += [0]
        else:
            if passenger['Age'] < 10 and passenger["Pclass"] <= 2:
                predictions += [1]
            elif passenger['Fare'] > 500:
                predictions += [1]
            else:
                predictions += [0]

    return predictions

## Loading data
in_file = 'titanic_data.csv' ### Loading the corpus
df = pd.read_csv(in_file)

#print(df.head())


### Remvoing the "Survived" outcome to a list
outcomes = df['Survived']
data = df.drop('Survived',axis=1)

## Data without "survived" field
#print (data.head())

### Tesing accuracy score function
predictions = pd.Series(np.ones(5, dtype = int))
print accuracy_score(outcomes[:5], predictions)
"""
### Converting dataframe to dict
data_dict = data.set_index('Name').T.to_dict('list')
data_keys = data_dict.keys()
print(data_dict[data_keys[0]])
"""

### Making predictions
print ("Prediction Model 0: No one survives",accuracy_score(outcomes,prediction_0(data)))
print ("Prediction Model 1: Female survive", accuracy_score(outcomes,prediction_1(data)))
print ("Prediction Model 2: Male below age 10 year  survive", accuracy_score(outcomes,prediction_2(data)))
print ("Prediction Model 3: Accuracy more than 80%", accuracy_score(outcomes,prediction_3(data)))

