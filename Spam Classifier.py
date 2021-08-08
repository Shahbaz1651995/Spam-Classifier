# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 19:43:45 2021

@author: MS076027
"""

####  Spam Classifier
import pandas as pd
messages = pd.read_csv("C:\\Users\\MS076027\\OneDrive - Cerner Corporation\\Documents\\Study material\\DataScience Directory\\Text mining\\Krish Naik Classes\\smsspamcollection\\SMSSpamCollection",delimiter='\t',names=["label","message"])

messages


### data Cleaning and preprocessing
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lz = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lz.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values


## train_test_split for model building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state = 0)

## training model using Naive bayes 
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)


##confussion matrix to check accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


###accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)







