# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init
import plotly.express as px
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r'C:\Users\PC\Downloads\Dataset\sentiment_analysis.csv')
print(df.head())

# Function to check for null values and data types
def null_count(df):
    return pd.DataFrame({
        'features': df.columns,
        'dtypes': df.dtypes.values,
        'NaN count': df.isnull().sum().values,
        'NaN percentage': df.isnull().sum().values / df.shape[0]
    }).style.background_gradient(cmap='Set3', low=0.1, high=0.01)

print(null_count(df))

# Check for duplicates
print(f"Number of duplicated rows: {df.duplicated().sum()}")

# Display columns
print(df.columns)

# Print number of distinct values for each column
for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")

# Feature engineering
# Drop irrelevant columns
columns_to_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'Hashtags', 'Day', 'Hour', 'Sentiment']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')


# Clean up 'Platform' and 'Country' columns
# Print column names to debug
print("Columns in DataFrame:", df.columns)

# Proceed with operations if 'Country' column exists
if 'Country' in df.columns:
    df['Country'] = df['Country'].str.strip()
else:
    print("'Country' column does not exist in the DataFrame.")

# Proceed with operations if 'Time of Tweet' column exists
# Define a mapping from time categories to approximate hours
time_mapping = {
    'morning': '06:00:00',  # Assuming morning is from 6 AM
    'noon': '12:00:00',    # Noon is 12 PM
    'night': '18:00:00'    # Night is 6 PM
}

# Check if 'Time of Tweet' column exists and is valid
if 'Time of Tweet' in df.columns:
    # Convert 'Time of Tweet' based on the mapping
    df['Time of Tweet'] = df['Time of Tweet'].map(time_mapping)

    # Ensure you have a 'Date' column or create one if needed
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime('today').strftime('%Y-%m-%d')  # Placeholder date

    # Combine date with time
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time of Tweet'], errors='coerce')

    # Drop rows where DateTime conversion failed
    df = df.dropna(subset=['DateTime'])
else:
    print("Column 'Time of Tweet' not found in DataFrame.")

# Print the updated DataFrame
print(df.head())
if 'Time of Tweet' in df.columns:
    df['Time of Tweet'] = pd.to_datetime(df['Time of Tweet'])
    df['Day_of_Week'] = df['Time of Tweet'].dt.day_name()
else:
    print("'Time of Tweet' column does not exist in the DataFrame.")

# Map month numbers to names
month_mapping = {
    1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
    5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
    9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
}
df['Month'] = df['Time of Tweet'].dt.month.map(month_mapping)
df['Month'] = df['Month'].astype('object')

# Text cleaning and preprocessing function
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

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

# Apply cleaning function to 'Text' column
df["Clean_Text"] = df["Text"].apply(clean)

# Vectorize text data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Clean_Text'])

# Prepare labels
y = df['sentiment']
# Print the column names to verify
print("Columns in DataFrame:", df.columns)

# Check if 'Sentiment' column exists and retrieve it
if 'sentiment' in df.columns:
    y = df['sentiment']
    print("Successfully retrieved 'sentiment' column.")
else:
    print("Column 'sentiment' not found in DataFrame.")
    # Handle the missing column scenario
    y = None  # or set y to an appropriate default value
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save vectorizer and model to .pkl files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("vectorizer.pkl and sentiment_model.pkl have been created and saved successfully.")

# Display unique values in specified columns
specified_columns = ['Platform', 'Country', 'Year', 'Month', 'Day_of_Week']
for col in specified_columns:
    total_unique_values = df[col].nunique()
    print(f'Total unique values for {col}: {total_unique_values}')
    
    top_values = df[col].value_counts()

    # Print top values with color formatting
    colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]
    for i, (value, count) in enumerate(top_values.items()):
        color = colors[i % len(colors)]
        print(f'{color}{value}: {count}{Fore.RESET}')

    print('\n' + '=' * 30 + '\n')

# Create a copy of the DataFrame for sentiment analysis
df1 = df.copy()

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Apply sentiment analysis
df1['Vader_Score'] = df1['Clean_Text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])
df1['Sentiment'] = df1['Vader_Score'].apply(lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral'))

# Display sentiment analysis results
print(df1[['Clean_Text', 'Vader_Score', 'Sentiment']].head())

# Plot sentiment distribution
colors = ['#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0)

sentiment_counts = df1['Sentiment'].value_counts()

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct=lambda p: f'{p:.2f}%\n({int(p * sum(sentiment_counts) / 100)})',
    wedgeprops=dict(width=0.7),
    textprops=dict(size=10, color="r"),
    pctdistance=0.7,
    colors=colors,
    explode=explode,
    shadow=True
)

# Add a center circle for better visualization
center_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
fig.gca().add_artist(center_circle)

ax.text(0, 0, 'Sentiment\nDistribution', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
ax.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
ax.axis('equal')

plt.show()

# Plot count of sentiments by year
plt.figure(figsize=(12, 6))
sns.countplot(x='Year', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Years and Sentiment')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot count of sentiments by month
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Month and Sentiment')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot count of sentiments by day of week
plt.figure(figsize=(12, 6))
sns.countplot(x='Day_of_Week', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Day of Week and Sentiment')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot count of sentiments by platform
plt.figure(figsize=(12, 6))
sns.countplot(x='Platform', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Platform and Sentiment')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot count of sentiments by top 10 countries
plt.figure(figsize=(12, 6))
top_10_countries = df1['Country'].value_counts().head(10).index
df_top_10_countries = df1[df1['Country'].isin(top_10_countries)]

sns.countplot(x='Country', hue='Sentiment', data=df_top_10_countries, palette='Paired')
plt.title('Relationship between Country and Sentiment (Top 10 Countries)')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Generate and display a bar chart of common words
df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])

fig = px.bar(top_words_df,
            x="count",
            y="Common_words",
            title='Common Words in Text Data',
            orientation='h',
            width=700,
            height=700,
            color='Common_words')

fig.show()
