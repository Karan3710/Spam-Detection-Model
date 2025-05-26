import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    return " ".join([word for word in words if word not in stop_words])

def train_and_save_model():
    df = pd.read_csv(r"C:\Users\karan\Downloads\archive (11)\spam.csv", encoding='latin-1')[['v1', 'v2']]
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

def load_model_and_vectorizer():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

def predict_spam(text, model, tfidf):
    cleaned = preprocess_text(text)
    transformed = tfidf.transform([cleaned])
    prediction = model.predict(transformed)[0]
    return "SPAM" if prediction == 1 else "HAM"

if __name__ == "__main__":
    train_and_save_model()  # Uncomment this line to train and save the model
    