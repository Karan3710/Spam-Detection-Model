# Spam-Detection-Model
# 📧 Spam Mail Detection using NLP and Machine Learning

This project is a Python-based spam detection system that uses Natural Language Processing (NLP) and a supervised learning model (Multinomial Naive Bayes) to classify SMS messages as **SPAM** or **HAM** (not spam).

## 🚀 Features
- Preprocesses raw text (lowercase, punctuation removal, stopword filtering)
- Converts messages to numerical form using TF-IDF vectorization
- Trains a Naive Bayes classifier
- Saves and loads the model and vectorizer using `pickle`
- Simple prediction function for new text messages

## 📁 Files
- `spam_detector.py`: Core script to preprocess, train, and predict
- `model.pkl`: Trained spam detection model (generated after training)
- `vectorizer.pkl`: Fitted TF-IDF vectorizer (generated after training)
- `spam.csv`: SMS Spam Collection dataset (UCI ML repository)

## 🛠 Requirements

Install dependencies using pip:
```bash
pip install -r requirements.txt
