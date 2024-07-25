# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from collections import Counter

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model and vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('sentiment_model.pkl', 'rb'))

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

# Streamlit app layout
st.title('Sentiment Analysis App')

# Text input for prediction
st.subheader("Predict Sentiment of Your Text")
user_input = st.text_area("Enter your text here:")

if user_input:
    # Clean the input text
    def clean(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = " ".join(text.split())
        tokens = word_tokenize(text)
        cleaned_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
        cleaned_text = ' '.join(cleaned_tokens)
        return cleaned_text

    cleaned_input = clean(user_input)
    
    # Vectorize and predict sentiment
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]
    
    # Display the prediction
    st.write(f"Predicted Sentiment: {prediction.capitalize()}")
    
    # Display sentiment score using VADER
    vader_score = analyzer.polarity_scores(user_input)['compound']
    sentiment_label = 'positive' if vader_score >= 0.05 else ('negative' if vader_score <= -0.05 else 'neutral')
    st.write(f"VADER Sentiment Score: {vader_score:.2f}")
    st.write(f"VADER Sentiment Label: {sentiment_label.capitalize()}")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(df.head())

    # Function to clean text
    def clean(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = " ".join(text.split())
        tokens = word_tokenize(text)
        cleaned_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
        cleaned_text = ' '.join(cleaned_tokens)
        return cleaned_text

    # Clean text data
    if 'Text' in df.columns:
        df["Clean_Text"] = df["Text"].apply(clean)
        st.write("Cleaned text preview:")
        st.write(df[['Text', 'Clean_Text']].head())
    else:
        st.write("'Text' column is missing in the uploaded CSV file.")

    # Feature engineering
    if 'Time of Tweet' in df.columns:
        time_mapping = {
            'morning': '06:00:00',
            'noon': '12:00:00',
            'night': '18:00:00'
        }
        df['Time of Tweet'] = df['Time of Tweet'].map(time_mapping)
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime('today').strftime('%Y-%m-%d')
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time of Tweet'], errors='coerce')
        df = df.dropna(subset=['DateTime'])
        df['Time of Tweet'] = pd.to_datetime(df['Time of Tweet'])
        df['Day_of_Week'] = df['Time of Tweet'].dt.day_name()
        month_mapping = {
            1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
            5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
            9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
        }
        df['Month'] = df['Time of Tweet'].dt.month.map(month_mapping)
        df['Month'] = df['Month'].astype('object')

    # Sentiment analysis
    if 'Clean_Text' in df.columns:
        X = vectorizer.transform(df['Clean_Text'])
        if 'sentiment' in df.columns:
            y = df['sentiment']
            df['Predicted_Sentiment'] = model.predict(X)
            st.write("Predicted Sentiments:")
            st.write(df[['Clean_Text', 'Predicted_Sentiment']].head())
        else:
            st.write("'Sentiment' column is missing in the uploaded CSV file.")
    else:
        st.write("'Clean_Text' column is missing in the uploaded CSV file.")

    # Visualization
    st.subheader("Sentiment Distribution")
    if 'Predicted_Sentiment' in df.columns:
        sentiment_counts = df['Predicted_Sentiment'].value_counts()
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct=lambda p: f'{p:.2f}%\n({int(p * sum(sentiment_counts) / 100)})',
            wedgeprops=dict(width=0.7),
            textprops=dict(size=10, color="r"),
            pctdistance=0.7,
            colors=['#66b3ff', '#99ff99', '#ffcc99'],
            explode=(0.1, 0, 0),
            shadow=True
        )
        center_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
        fig.gca().add_artist(center_circle)
        ax.text(0, 0, 'Sentiment\nDistribution', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
        ax.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax.axis('equal')
        st.pyplot(fig)

    # Additional plots
    st.subheader("Sentiment by Year")
    if 'Year' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Year', hue='Predicted_Sentiment', data=df, palette='Paired')
        plt.title('Relationship between Years and Sentiment')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot()

    st.subheader("Sentiment by Month")
    if 'Month' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Month', hue='Predicted_Sentiment', data=df, palette='Paired')
        plt.title('Relationship between Month and Sentiment')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot()

    st.subheader("Sentiment by Day of Week")
    if 'Day_of_Week' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Day_of_Week', hue='Predicted_Sentiment', data=df, palette='Paired')
        plt.title('Relationship between Day of Week and Sentiment')
        plt.xlabel('Day of Week')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot()

    st.subheader("Sentiment by Platform")
    if 'Platform' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Platform', hue='Predicted_Sentiment', data=df, palette='Paired')
        plt.title('Relationship between Platform and Sentiment')
        plt.xlabel('Platform')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot()

    st.subheader("Sentiment by Top 10 Countries")
    if 'Country' in df.columns:
        top_10_countries = df['Country'].value_counts().head(10).index
        df_top_10_countries = df[df['Country'].isin(top_10_countries)]
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Country', hue='Predicted_Sentiment', data=df_top_10_countries, palette='Paired')
        plt.title('Relationship between Country and Sentiment (Top 10 Countries)')
        plt.xlabel('Country')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot()

    st.subheader("Common Words")
    if 'Clean_Text' in df.columns:
        df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
        top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
        top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])
        fig = px.bar(top_words_df, x="count", y="Common_words", title='Common Words in Text Data', orientation='h', width=700, height=700, color='Common_words')
        st.plotly_chart(fig)
