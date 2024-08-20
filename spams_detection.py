# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 08:53:54 2022

@author: Jabbouri Mohamed
"""

#Importation de dataset :
    
import pandas as pd
dataset = pd.read_csv("dataset_spams.csv")
dataset["text"] = [word.replace("Subject: ",'') for word in dataset["text"]]


#To lower
dataset["text"] = dataset["text"].str.lower()

#remove punctuation
import string
punc = list(string.punctuation)
def remove_punc(text):
    for i in string.punctuation:
        text = text.replace(i, '')
    return text
dataset["text"] = dataset["text"].apply(remove_punc)

#tokenisation
import nltk
dataset["text"] = dataset["text"].apply(nltk.word_tokenize)

#lemmatization 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def word_lemmatization(words):
    new_words = []
    for word in words:
        new_word = lemmatizer.lemmatize(word)
        new_words.append(new_word)
    return new_words
dataset["text"] = dataset["text"].apply(word_lemmatization)

#Remove stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(words):
    stop_words = stopwords.words('english')
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words

dataset["text"] = dataset["text"].apply(remove_stopwords)


dataset["text"] = [' '.join(word) for word in dataset["text"]]

#Spliting data into train set ad test set 
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset["text"], 
                                                                    dataset["spam"], 
                                                                    test_size=0.25,
                                                                    random_state = 0)


#Encodage
from sklearn.preprocessing import LabelEncoder

Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.transform(y_test)



#Vectorization 

from sklearn.feature_extraction.text import TfidfVectorizer

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(dataset["text"])
Train_X_Tfidf = Tfidf_vect.transform(X_train)
Test_X_Tfidf = Tfidf_vect.transform(X_test)


#training and predecting using Naive Bayes :
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
import time

my_model_nb = naive_bayes.MultinomialNB()
start_nb = time.time()
my_model_nb.fit(Train_X_Tfidf, y_train)
nb_time = time.time() - start_nb

prediction = my_model_nb.predict(Test_X_Tfidf)
model_name = my_model_nb.__class__.__name__
print(f"The accuracy using a {model_name} is { accuracy_score(prediction, y_test)*100:.3f}"
      f" with a fitting time of {nb_time:.3f} seconds "
      )

#training and predecting using LogisticRegression :
from sklearn.linear_model import LogisticRegression

my_model_lr = LogisticRegression()
start_lr = time.time()
my_model_lr.fit(Train_X_Tfidf, y_train)
lr_time = time.time() - start_lr

prediction = my_model_lr.predict(Test_X_Tfidf)
model_name = my_model_lr.__class__.__name__
print(f"The accuracy using a {model_name} is { accuracy_score(prediction, y_test)*100:.3f}"
      f"with a fitting time of {lr_time:.3f} seconds "
      )

#training and predecting using KNN :
from sklearn.neighbors import KNeighborsClassifier

my_model_knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn_start = time.time()
my_model_knn.fit(Train_X_Tfidf, y_train)
knn_time = time.time() - knn_start

prediction = my_model_knn.predict(Test_X_Tfidf)
model_name = my_model_knn.__class__.__name__
print(f"The accuracy using a {model_name} is { accuracy_score(prediction, y_test)*100:.3f}"
      f"with a fitting time of {knn_time:.3f} seconds "
      )

def spam_test(text,my_model) :
    text = text.lower()
    text = remove_punc(text)
    lis_text = nltk.word_tokenize(text)
    lis_text= word_lemmatization(lis_text)
    lis_text= remove_stopwords(lis_text)
    text = ' '.join(lis_text)
   
    text=Tfidf_vect.transform([text])
    pred = my_model.predict(text)
    
    if pred == 1 :
        print("It's a spam")
    if pred == 0 :
        print("It's not a spam")
        

# Save the models as .pkl files
import pickle
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(my_model_lr, f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(my_model_knn, f)

with open('nb_model.pkl', 'wb') as f:
    pickle.dump(my_model_nb, f)

        



    
    











