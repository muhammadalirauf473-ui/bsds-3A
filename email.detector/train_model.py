import os
import ssl
import pandas as pd
import urllib.request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Bypass MacOS SSL certificate verification errors for Python's urllib
ssl._create_default_https_context = ssl._create_unverified_context

def download_data():
    """Downloads a public SMS spam dataset if it doesn't exist."""
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    filename = "spam_dataset.tsv"
    
    if not os.path.exists(filename):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    return filename

def train():
    filename = download_data()
    
    print("Loading data...")
    # The dataset has two columns: label ('ham' or 'spam') and message
    df = pd.read_csv(filename, sep='\t', header=None, names=['label', 'message'])
    
    # Map labels to 0 (ham/safe) and 1 (spam)
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    
    X = df.message
    y = df.label_num
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the Multinomial Naive Bayes model...")
    # Create a pipeline that first extracts TF-IDF features then trains the classifier
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe (Ham)', 'Spam']))
    
    # Save the trained pipeline
    model_filename = 'spam_model.pkl'
    joblib.dump(pipeline, model_filename)
    print(f"\nModel successfully saved to {model_filename}")

if __name__ == '__main__':
    train()
