import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# used by lemmatizer
#nltk.download('wordnet')
#nltk.download('omw-1.4')

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
stopWords = set(stopwords.words('english'))

# takes input list and removes unwanted symbols and stop words

def clean(set):
    set = regexClean(set)
    set = stopClean(set)
    set = porterStemClean(set)
        
    return set

# compiles regex pattern strings to look for unwanted markup tags and punctuation
def regexClean(set):
    set = [re.compile("[.;:!\'?,\"()\[\]]").sub("", line.lower()) for line in set]
    set = [re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)").sub(" ", line) for line in set]
    return set

# removes words with little to no meaning (i.e. in, of, at, etc.)
def stopClean(set):
    stopSet=[]
    tempSet=[]
    for review in set:
        tempSet = [word for word in review.split() if word not in stopWords]
        stopSet.append(' '.join(tempSet))
    return stopSet

# stems words so as to chop off the last few characters (stem); may lead to incorrect interpretation
# least aggressive algorithm
def porterStemClean(set):
    stemmer = PorterStemmer()
    stemSet = [' '.join([stemmer.stem(word) for word in review.split()]) for review in set]
    return stemSet

regexTrainingSet = regexClean(trainingSet)
regexTestSet = regexClean(testSet)

stopTrainingSet = stopClean(regexTrainingSet)
stopTestSet = stopClean(regexTestSet)

lemTrainingSet = porterStemClean(stopTrainingSet)
lemTestSet = porterStemClean(stopTestSet)


ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(lemTrainingSet)
X = ngram_vectorizer.transform(lemTrainingSet)
X_test = ngram_vectorizer.transform(lemTestSet)

# target is first 12500 should be positive (1) and second 12500 should be negative (0)
target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

#for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
#    lr = LinearSVC(C=c, max_iter=900)
#    lr.fit(X_train, y_train)
#    print ("Accuracy for C=%s: %s" 
#           % (c, accuracy_score(y_val, lr.predict(X_val))))

#for c in [1, 1.25, 1.5, 1.75, 2]:
    
#    lr = LinearSVC(C=c, max_iter=500)
#    lr.fit(X_train, y_train)
#    print ("Accuracy for C=%s: %s" 
#           % (c, accuracy_score(y_val, lr.predict(X_val))))

# c = 0.01 has been found to yield the highest accuracy for this model, about 0.88912

final_ngram = LinearSVC(C=0.01, max_iter=900)
final_ngram.fit(X, target)
print ("Final Accuracy: %s" 
    % accuracy_score(target, final_ngram.predict(X_test)))

# Final Accuracy: 0.8876; 0.836% increase over basic implementation
