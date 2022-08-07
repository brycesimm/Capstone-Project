import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# reading in training set and removing spaces between reviews
trainingSet = []
for line in open('trainSet.txt', 'r', encoding="utf8"):
    trainingSet.append(line.strip())
# reading in testing set
testSet = []
for line in open('testSet.txt', 'r', encoding="utf8"):
    testSet.append(line.strip())

#nltk.download('stopwords')
#nltk.download('punkt')
#stopWords = set(stopwords.words('english'))
stopWords = ['in', 'of', 'at', 'a', 'the', 'an', 'as', 'to', 'or']

# compiles regex pattern strings to look for unwanted markup tags and punctuation
def regexClean(set):
    set = [re.compile("[.;:!\'?,\"()\[\]]").sub("", line.lower()) for line in set]
    set = [re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)").sub(" ", line) for line in set]
    return set

# stems words so as to chop off the last few characters (stem); may lead to incorrect interpretation
# least aggressive algorithm
def porterStemClean(set):
    stemmer = PorterStemmer()
    stemSet = [' '.join([stemmer.stem(word) for word in review.split()]) for review in set]
    return stemSet

regexTrainingSet = regexClean(trainingSet)
regexTestSet = regexClean(testSet)

stemTrainingSet = porterStemClean(regexTrainingSet)
stemTestSet = porterStemClean(regexTestSet)

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stopWords)
ngram_vectorizer.fit(stemTrainingSet)

# target is first 12500 should be positive (1) and second 12500 should be negative (0)
target = [1 if i < 12500 else 0 for i in range(25000)]

X = ngram_vectorizer.transform(stemTrainingSet)
X_test = ngram_vectorizer.transform(stemTestSet)

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    
    svm = LogisticRegression(C=c)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, svm.predict(X_val))))

final_ngram = LogisticRegression(C=0.1, max_iter=500)
final_ngram.fit(X, target)
print ("Final Accuracy: %s" 
    % accuracy_score(target, final_ngram.predict(X_test)))

# Final Accuracy: 0.89956;

