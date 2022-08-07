# Bag of Words Vectorization implementation using Logistic Regression to predict value of positive or negative. 
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


# reading in training set and removing spaces between reviews
trainingSet = []
for line in open('trainSet.txt', 'r', encoding="utf8"):
    trainingSet.append(line.strip())
# reading in testing set
testSet = []
for line in open('testSet.txt', 'r', encoding="utf8"):
    testSet.append(line.strip())

# compiles regex pattern strings to look for unwanted markup tags and punctuation
# takes input array and replaces all occurrences of unwanted symbols with a space or no space
def clean(set):
    set = [re.compile("[.;:!\'?,\"()\[\]]").sub("", line.lower()) for line in set]
    set = [re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)").sub(" ", line) for line in set]
    return set

# clean both sets
cleanTrainingSet = clean(trainingSet)
cleanTestSet = clean(testSet)

# CountVectorizer is used to create an array of 1's and 0's (vector)
# 1 corresponds to a word's appearance and 0 corresponds to no appearance
vectorizer = CountVectorizer(binary=True)
# Bag of Words - each review in the set is transformed into a binary encoded string 
# A 1 in the string means the word at that index in the dictionary occurred in the review
# A 0 in the string means the word at that index in the dictionary did not occur in the review
# Creates a dictionary of tokens from the training set and extracts token (word) counts in 
# iterable set using fitted dictionary
trainBag = vectorizer.fit_transform(cleanTrainingSet) # 25000 elements
# Very sparse matrix
#for item in trainBag.toarray():
#    print(item)
# test set is transformed as well
testBag = vectorizer.transform(cleanTestSet) # 25000 elements

# target is first 12500 should be positive (1) and second 12500 should be negative (0)
target = [1 if i < 12500 else 0 for i in range(25000)]

# takes *arrays and splits each of them in half, returning list of 2*len(arrays) elements
# train_size is the proportion of the dataset to include in the train split
X_train, X_trainTest, Y_train, Y_trainTest = train_test_split(trainBag, target, train_size = 0.75)

# c is the regularization hyperparameter to be tuned for LR
# regularization shrinks coefficients in the regression equation 
# to reduce variance in the model. This is to prevent overfit. 
# Overfit leads to the model being unable to be applied to different 
# data sets (failing to generalize). So, without regularization, a model
# may perform well on one set (like training set), but fail on another (test set)
# Regularization lowering variance can help deal with unseen data, which is 
# plausible to occur in human-written reviews based on experience. 

#for c in np.arange(-5,5):
#    lr = LogisticRegression(C=(10.0**c))
#    lr.fit(X_train, Y_train)
#    print ("Accuracy for C=%s: %s" % (c, accuracy_score(Y_trainTest, lr.predict(X_trainTest))))

#for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
#    lr = LogisticRegression(C=c)
#    lr.fit(X_train, Y_train)
#    print ("Accuracy for C=%s: %s" % (c, accuracy_score(Y_trainTest, lr.predict(X_trainTest))))

# c = 0.1 has been found to yield the highest accuracy for this model, about 0.88

lr = LogisticRegression(C=0.1)
lr.fit(trainBag, target)
print("Test Accuracy: %s" % accuracy_score(target, lr.predict(testBag)))

# Test Accuracy: 0.87924

# Grabs feature names and their coefficients and stores them as tuples in a list
tuples = list(zip(vectorizer.get_feature_names_out(), lr.coef_[0]))
# Creates dictionary using list of tuples
featureCoefDict = dict(tuples)

reverseSortedKeys = sorted(featureCoefDict, key=featureCoefDict.get, reverse=True)
mostPositive = {}
i=0
for w in reverseSortedKeys:
    if i<5:
        mostPositive[w]=featureCoefDict[w]
        i=i+1
print (mostPositive)

#{'excellent': 1.0481065091451305, 
# 'perfect': 0.8991050807846164, 
# 'refreshing': 0.7371016160324548, 
# 'superb': 0.7345532780104588, 
# 'wonderfully': 0.7201715993636091}

SortedKeys = sorted(featureCoefDict, key=featureCoefDict.get)
mostNegative = {}
i=0
for w in SortedKeys:
    if i<5:
        mostNegative[w]=featureCoefDict[w]
        i=i+1
print (mostNegative)

#{'worst': -1.544527936053958, 
# 'waste': -1.3816850174611393, 
# 'awful': -1.182003547792695, 
# 'poorly': -1.0957790775175307, 
# 'disappointment': -1.072719321108437}