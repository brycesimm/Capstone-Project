import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
#stopWords = set(stopwords.words('english'))
stopWords = ['in', 'of', 'at', 'a', 'the', 'an', 'as', 'to', 'or']


# compiles regex pattern strings to look for unwanted markup tags and punctuation
def regexClean(set):
    set = [re.compile("[.;:!\'?,\"()\[\]]").sub("", line.lower()) for line in set]
    set = [re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)").sub(" ", line) for line in set]
    return set

# alternative normalization to stemming which considers context and converts word to base form; aka lemma
def lemmaClean(set):
    lemmatizer = WordNetLemmatizer()
    lemmaSet = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in set]
    return lemmaSet

regexTrainingSet = regexClean(trainingSet)
regexTestSet = regexClean(testSet)

lemTrainingSet = lemmaClean(regexTrainingSet)
lemTestSet = lemmaClean(regexTestSet)


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stopWords)
tfidf_vectorizer.fit(lemTrainingSet)
X = tfidf_vectorizer.transform(lemTrainingSet)
X_test = tfidf_vectorizer.transform(lemTestSet)

# target is first 12500 should be positive (1) and second 12500 should be negative (0)
target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [3, 3.5, 4, 4.5, 5]:
    
    svm = LogisticRegression(C=c)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, svm.predict(X_val))))

final_tfidf = LogisticRegression(C=5, max_iter=500)
final_tfidf.fit(X, target)
print ("Final Accuracy: %s" 
    % accuracy_score(target, final_tfidf.predict(X_test)))

# Final Accuracy: 0.89256;
