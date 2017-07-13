import string
from collections import Counter ### Gives the frequency of words in the list passed
from sklearn.feature_extraction.text import CountVectorizer


### Definitions ###

### making data lower case
def lower_case(data):
    for i in range(len(data)):
        data[i] = data[i].lower()

### Removing Punctuation
def rem_puct(data):
    ret_list = []
    for i in data:
        ret_list += [i.translate(string.maketrans('', ''), string.punctuation)]

    return ret_list

### Taking Data and labels out
def DataFormat(data):
    data_in = []
    labels = []
    for item in data:
        data_in += [item[1]]
        labels += [item[0]]
    return data_in,labels

### Tokenizing the data
def token_words(data):
    for i in range(len(data)):
        data[i] = data[i].split()

### Counting
def count_words(data):
    frequency_list = []
    for item in data:
        frequency_count = Counter(item)
        frequency_list += [frequency_count]
    return frequency_list


fp = open("SMSSpamCollection","r")
data = fp.readlines()
for i in range(len(data)):
    data[i] = data[i].split()

names = []
print ("Number of messages %d" %len(data))
for item in data:
    if item[0] == "ham":
        names += [[0,' '.join(item[1:])]]
    else:
        names += [[1,' '.join(item[1:])]]


### Formating data and labels in seperate lists
data_in,labels = DataFormat(names)

#Step 2.2: Implementing Bag of Words from scratch

### Step 1: Making data lower case
lower_case(data_in)

### Step 2: Removing puctuations
sans_punctuation_names = rem_puct(data_in)

### Step 3; Tokenization
token_words(sans_punctuation_names)
preprocessed_data = sans_punctuation_names

## Step 4: Counting the frequency of words
frequency_list = count_words(preprocessed_data)

