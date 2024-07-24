# train_save_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Sample dataset
data = {
    'Text': ["I love this product", "This is the worst thing I bought", "Absolutely fantastic", "Not good at all"],
    'Sentiment': [1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])
y = df['Sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and the vectorizer
joblib.dump(model, 'C:/Users/PC/Downloads/Sentiment model/sentiment_model.pkl')
joblib.dump(vectorizer, 'C:/Users/PC/Downloads/Sentiment model/vectorizer.pkl')

print("Model and vectorizer saved successfully.")
