# Bag of Words Vectorization implementation using Logistic Regression to predict value of positive or negative. 
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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
vectorizer = CountVectorizer(binary=True, ngram_range=(1,2))
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

mnb = MultinomialNB()
mnb.fit(trainBag, target)
print("Test Accuracy: %s" % accuracy_score(target, mnb.predict(testBag)))

# Test Accuracy: 0.86568
