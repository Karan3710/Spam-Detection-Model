
import os
import string
import pickle
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    return " ".join([word for word in words if word not in stop_words])

# Train model and save artifacts
def train_and_save_model():
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    df['cleaned_message'] = df['message'].apply(preprocess_text)

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['cleaned_message'])
    y = df['label_num']

    model = MultinomialNB()
    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

# Load model and vectorizer from disk
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

# Predict function
def predict_spam(text, model, tfidf):
    cleaned = preprocess_text(text)
    transformed = tfidf.transform([cleaned])
    prediction = model.predict(transformed)[0]
    return "SPAM" if prediction == 1 else "HAM"

# Check if model exists, train if not
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    train_and_save_model()

# Streamlit UI
st.set_page_config(page_title="📨 Spam Mail Detection", page_icon="📩")

st.title("📨 Spam Mail Detection App")
st.markdown("Enter a message below to check if it's **Spam** or **Ham (Not Spam)**.")

user_input = st.text_area("✉️ Enter your message here:", height=150)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        model, tfidf = load_model_and_vectorizer()
        result = predict_spam(user_input, model, tfidf)
        if result == "SPAM":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is HAM (Not Spam).")
