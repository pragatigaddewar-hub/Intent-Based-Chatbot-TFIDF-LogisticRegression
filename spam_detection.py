## Import Libraries
import os
import re
import numpy as np
import pandas as pd

## Folder Path (spam.csv and this file are in the same folder)
folder = os.path.dirname(os.path.abspath(__file__))

## Load Data
df = pd.read_csv(os.path.join(folder,'spam.csv'), encoding='latin-1', usecols=[0,1])

## Count Messages
print("Shape :" ,df.shape)
print("Total Messages :" ,len(df))

## Rename Columns
df.columns = ['label','message']

print("Shape :" ,df.shape)

## Class Count (ham and spam)
print(df['label'].value_counts())

## Convert Labels Into Numbers (spam = 1, ham = 0)
df['label'] = df['label'].map({'spam':1,'ham':0})

## Text Cleaning Function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation, numbers and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    return text

## Clean All Messages
df['clean_message'] = df['message'].apply(clean_text)

## Split Data (4457 train, 1115 test)
from sklearn.model_selection import train_test_split

# stratify keeps the same spam ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

## TF-IDF (Top 3000 Words, Remove Stopwords)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000, stop_words='english')

# Fit and transform on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Only transform on test data (avoids data leakage)
X_test_tfidf = tfidf.transform(X_test)

print("Train Shape :" ,X_train_tfidf.shape)
print("Test Shape :" ,X_test_tfidf.shape)

## Create And Train Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

## Prediction
y_pred = model.predict(X_test_tfidf)

## Accuracy
from sklearn.metrics import accuracy_score

print("Test Accuracy :" ,accuracy_score(y_test, y_pred))

## Classification Report
from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred, target_names=['ham','spam'])
print("Classification Report :")
print(cr)

## Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :")
print(cm)

## Predict New Message
new_message = "Congratulations! You won a free prize, call now to claim"

# Clean and convert the new message with the same TF-IDF
new_message_tfidf = tfidf.transform([clean_text(new_message)])

print("Prediction :" ,model.predict(new_message_tfidf))