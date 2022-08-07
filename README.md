# Capstone-Project

This repository includes every sentiment analysis test I have attempted for my Senior Capstone Project in Spring 2022.
The project utililzes of 50,000 movie reviews mined from IMDb by researchers at Stanford University 
(See their published paper at: https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

This project was intended to be an introduction for myself to explore a new field of Computer Science. I utilized Python and libraries such as sklearn and nltk to perform Sentiment Analysis using Natural Language Processing methods in order to distinguish which reviews were positive and which were negative. The reviews are split evenly into two .txt files, one for training the classifiers and one for testing the classifiers. Various forms of pre-processing were tested, including the removal of stop words and non-english text, stemming, and lemmatizing. Two feature extration methods were used: Bag of Words and TF-IDF. Classifiers used included Logistic Regression, Multinomial Naive Bayes, Stochastic Gradient Descent, and LinearSVC. 

The first "base case" test I ran is contained in "LR_BOW_Basic.py" in which I used Logistic Regression, Bag of Words, automated stop word removal, and removal of non-english text using regex. This was used as the simplest implementation of Sentiment Analysis, only requiring tuning of one hyperparameter and minimal pre-processing. 

Subsequent files in the repository are the result of mixing and matching classifiers, feature extraction methods, and pre-processing methods to achieve the highest possible accuracy for the dataset I obtained. You will find there are 26 python files including the base case, which is only a portion of the maximum number of combinations I could have tested. This is because certain methods, such as the Lancaster stemming method, were counter-productive due to aggressiveness and yielded such low accuracy that alternative tests utilizing these methods were unnecessary. Therefore, the 26 python files show each of the avenues taken to try and find the "sweet spot" for analyzing sentiment with the highest possible accuracy. 

Each file is named with the classifier, vectorizer, and pre-processing method used.
Files with "Less_SW" prepended in the name are versions in which stop words are manually chosen rather than using
the nltk library. 

The Python files were written with version 3.10 installed, and require various imports from:
nltk, re, and sklearn

It is possible that nltk will require you to run a "nltk.download" command, and these should be commented
in the Python file(s) that require resources to be downloaded. 

Assuming all of the imports are installed and the correct version of Python is used, these files should
run on their own so long as testSet.txt and trainSet.txt are in the same directory. 

There is a spreadsheet named "Accuracies.xlsx" and it stores the accuracies each implementation
yielded. It also contains conditional formatting to highlight positive/negative changes in accuracy
over the baseline implementation, and highlights the most positive and most negative tests.

Additionally you will find the PDF document of the technical paper (IEEE format) I submitted at the conclusion of the capstone course, and it sums up my objective, methods, and results in greater detail. 
