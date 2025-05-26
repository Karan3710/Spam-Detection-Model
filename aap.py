import streamlit as st
from spam_detector import load_model_and_vectorizer, predict_spam

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
