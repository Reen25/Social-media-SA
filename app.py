# app.py
import streamlit as st
import joblib
import os

def load_model(file_path):
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        st.error(f"File not found: {file_path}")
        return None

def main():
    st.title("Sentiment Analysis App")

    model_path = "C:/Users/PC/Downloads/Sentiment model/sentiment_model.pkl"
    vectorizer_path = "C:/Users/PC/Downloads/Sentiment model/vectorizer.pkl"

    model = load_model(model_path)
    vectorizer = load_model(vectorizer_path)

    if model and vectorizer:
        user_input = st.text_area("Enter text for sentiment analysis")
        if st.button("Analyze"):
            X_example = vectorizer.transform([user_input])
            prediction = model.predict(X_example)
            st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
