# Intent Based Chatbot (TF-IDF + Logistic Regression)

A simple NLP chatbot that finds the **intent** of a user message and replies with a ready response.

## Dataset
- Small custom dataset (`intents.csv`), 40 messages
- 2 columns: message (user input) and intent (label)
- 4 intents: greeting, goodbye, thanks, help (10 messages each, balanced data)

## Project Workflow
1. Load data and clean text (lowercase, remove punctuation and numbers)
2. Train / test split: 32 train, 8 test (stratified)
3. TF-IDF vectorization (43 words in the training vocabulary, stopwords not removed because messages are very short)
4. Train a Logistic Regression model
5. Evaluate with accuracy and classification report
6. Chat loop: predict the intent of the message and print the matching reply

## Results
- Test accuracy: **0.75** (8 test messages, so one mistake changes accuracy by 12.5%)

| Intent   | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| goodbye  | 0.50      | 1.00   | 0.67     | 2       |
| greeting | 1.00      | 0.50   | 0.67     | 2       |
| help     | 1.00      | 0.50   | 0.67     | 2       |
| thanks   | 1.00      | 1.00   | 1.00     | 2       |

## Example Chat
```
You : see you
Intent : goodbye
Bot : Goodbye! Have a nice day.
You : help please
Intent : help
Bot : Sure, tell me what you need help with.
You : hi
Intent : greeting
Bot : Hi! How can I help you?
You : many thanks
Intent : thanks
Bot : You are welcome!
```

## How to Run
```
pip install -r requirements.txt
python chatbot.py
```
Keep `intents.csv` in the same folder as `chatbot.py`. Type `exit` to stop the chat.

## Project Structure
```
Intent-Based-Chatbot-TFIDF-LogisticRegression/
    intents.csv
    chatbot.py
    requirements.txt
    README.md
```

## Tools
Python, Pandas, Scikit-learn

## Author
Pragati Gaddewar
