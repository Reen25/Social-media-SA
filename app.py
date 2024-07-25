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

    model_path = "C:/Users/PC/Downloads/Dataset/sentiment_model.pkl"
    vectorizer_path = "C:/Users/PC/Downloads/Dataset/vectorizer.pkl"

    model = load_model(model_path)
    vectorizer = load_model(vectorizer_path)

    if model and vectorizer:
        user_input = st.text_area("Enter text for sentiment analysis")

        if st.button("Analyze"):
            if user_input:
                X_example = vectorizer.transform([user_input])
                prediction = model.predict(X_example)[0]

                # Map the prediction to a readable label
                sentiment_label = "Neutral"  # Default to neutral
                if prediction == 1:
                    sentiment_label = "Positive"
                elif prediction == -1:
                    sentiment_label = "Negative"
                elif prediction == 0:  # Assuming 0 represents neutral in your model
                    sentiment_label = "Neutral"

                st.write("Prediction:", sentiment_label)
            else:
                st.error("Please enter some text to analyze.")
    else:
        st.error("Model or vectorizer not found. Please check the file paths.")

if __name__ == "__main__":
    main()
