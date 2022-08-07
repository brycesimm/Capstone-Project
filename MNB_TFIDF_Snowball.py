import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
stopWords = set(stopwords.words('english'))

# takes input list and removes unwanted symbols and stop words

def clean(set):
    set = regexClean(set)
    set = stopClean(set)
    set = snowballStemClean(set)
        
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

# slightly improved version of porter stem; typically faster and more accurate
# slightly more aggressive algorithm
def snowballStemClean(set):
    stemmer = SnowballStemmer(language='english')
    stemSet = [' '.join([stemmer.stem(word) for word in review.split()]) for review in set]
    return stemSet

regexTrainingSet = regexClean(trainingSet)
regexTestSet = regexClean(testSet)

stopTrainingSet = stopClean(regexTrainingSet)
stopTestSet = stopClean(regexTestSet)

stemTrainingSet = snowballStemClean(stopTrainingSet)
stemTestSet = snowballStemClean(stopTestSet)


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_vectorizer.fit(stemTrainingSet)
X = tfidf_vectorizer.transform(stemTrainingSet)
X_test = tfidf_vectorizer.transform(stemTestSet)

# target is first 12500 should be positive (1) and second 12500 should be negative (0)
target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

final_ngram = MultinomialNB()
final_ngram.fit(X, target)
print ("Final Accuracy: %s" 
    % accuracy_score(target, final_ngram.predict(X_test)))

# Final Accuracy: 0.85652;

