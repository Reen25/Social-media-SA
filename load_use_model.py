# load_use_model.py
import joblib
import os

# Define the file paths
model_path = "C:/Users/PC/Downloads/Sentiment model/sentiment_model.pkl"
vectorizer_path = "C:/Users/PC/Downloads/Sentiment model/vectorizer.pkl"

# Load the model and vectorizer
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")
else:
    print(f"File not found: {model_path} or {vectorizer_path}")

# Example usage
example_text = ["I am very happy with this service"]
X_example = vectorizer.transform(example_text)
prediction = model.predict(X_example)
print("Prediction:", prediction)
