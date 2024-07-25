import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create a dummy vectorizer
vectorizer = CountVectorizer()
# Dummy training data
X_dummy = vectorizer.fit_transform(["sample text", "another text example"])
# Dummy labels
y_dummy = ["positive", "negative"]

# Create a dummy model
model = MultinomialNB()
model.fit(X_dummy, y_dummy)

# Save vectorizer to a .pkl file
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save model to a .pkl file
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
