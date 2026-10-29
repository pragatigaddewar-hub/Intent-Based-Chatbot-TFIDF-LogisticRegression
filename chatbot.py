## Import Libraries
import os
import re
import pandas as pd

## Folder Path (intents.csv and this file are in the same folder)
folder = os.path.dirname(os.path.abspath(__file__))

## Load Data (40 messages)
df = pd.read_csv(os.path.join(folder,'intents.csv'))

print("Shape :" ,df.shape)

## Class Count (4 intents)
print(df['intent'].value_counts())

## Text Cleaning Function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation, numbers and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    return text

## Clean All Messages
df['clean_message'] = df['message'].apply(clean_text)

## Split Data (32 train, 8 test)
from sklearn.model_selection import train_test_split

# stratify keeps the same intent ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'],
    df['intent'],
    test_size=0.2,
    random_state=42,
    stratify=df['intent']
)

## TF-IDF (stopwords are NOT removed, messages are very short)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

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

cr = classification_report(y_test, y_pred)
print("Classification Report :")
print(cr)

## Replies For Each Intent
replies = {
    'greeting' : "Hi! How can I help you?",
    'goodbye'  : "Goodbye! Have a nice day.",
    'thanks'   : "You are welcome!",
    'help'     : "Sure, tell me what you need help with."
}

## Chat With The Bot (type exit to stop)
print("Chatbot is ready. Type exit to stop.")

while True:

    # Take a message from the user
    user_message = input("You : ")

    # Stop the chat
    if user_message.lower() == 'exit':
        break

    # Clean and convert the message with the same TF-IDF
    user_tfidf = tfidf.transform([clean_text(user_message)])

    # Predict the intent
    intent = model.predict(user_tfidf)[0]

    print("Intent :" ,intent)
    print("Bot :" ,replies[intent])