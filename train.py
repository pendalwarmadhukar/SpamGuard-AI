import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import re
import string
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK data required
nltk.download('punkt')
nltk.download('stopwords')

print("Starting training pipeline...")

# 1. Download Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
zip_path = "smsspamcollection.zip"
data_path = "SMSSpamCollection"

if not os.path.exists(data_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Dataset downloaded and extracted.")
else:
    print("Dataset already found locally.")

# 2. Load dataset
df = pd.read_csv(data_path, sep='\t', names=['target', 'text'])

# 3. Preprocess
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df.drop_duplicates(keep='first', inplace=True)

print("Preprocessing text... this may take a moment.")
ps = PorterStemmer()
stop_words = stopwords.words('english')

def transform_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

# 4. Vectorization
print("Vectorizing Text data...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
print("Training the Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# 6. Save Model
print("Saving model and vectorizer...")
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("Done! You can now start the Streamlit app.")
