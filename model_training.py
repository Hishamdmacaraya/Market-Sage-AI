# %% [code] "model_training.py"
import re  # For regular expression operations
import string  # For string manipulation (punctuation removal)
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert text to TF-IDF features
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier for text classification
from sklearn.metrics import accuracy_score, classification_report  # For evaluating the classifier
import joblib  # For saving/loading models

def clean_text(text):
    """
    Clean input text by:
    - Removing URLs (using regex)
    - Removing punctuation
    - Converting to lowercase
    
    Args:
        text (str): The raw text to clean.
        
    Returns:
        str: The cleaned text.
    """
    # Remove any URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove punctuation using str.translate
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase for uniformity
    text = text.lower()
    return text

def load_and_preprocess_data(csv_path):
    """
    Load data from CSV and apply text cleaning.
    
    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        
    Returns:
        X: The cleaned text data.
        y: The corresponding sentiment labels.
    """
    df = pd.read_csv(csv_path)
    # Apply the cleaning function to each text entry
    df['cleaned_text'] = df['text'].apply(clean_text)
    X = df['cleaned_text']
    y = df['label']
    return X, y

def train_sentiment_model(csv_path="sample_data.csv", model_out="sentiment_model.pkl"):
    """
    Train a Naive Bayes classifier on the cleaned text data using TF-IDF features.
    Save the trained model and the TF-IDF vectorizer for later use.
    
    Args:
        csv_path (str): Path to the dataset CSV.
        model_out (str): Filename for the saved model.
    """
    # Load and preprocess data
    X, y = load_and_preprocess_data(csv_path)
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer on training data and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Transform the test data using the same vectorizer
    X_test_tfidf = vectorizer.transform(X_test)
    # Initialize and train the Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)
    
    # Evaluate the model on the test set
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test set: {:.2f}".format(accuracy))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the trained model and vectorizer using joblib
    joblib.dump(clf, model_out)
    joblib.dump(vectorizer, "vectorizer.pkl")
    print(f"[INFO] Model saved to {model_out}")
    print(f"[INFO] Vectorizer saved to vectorizer.pkl")

def load_sentiment_pipeline(model_path="sentiment_model.pkl", vectorizer_path="vectorizer.pkl"):
    """
    Load the saved sentiment model and vectorizer.
    
    Returns:
        A function that takes raw text as input and returns the predicted sentiment label.
    """
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    def predict_sentiment(text):
        # Clean the input text and convert it to TF-IDF features
        cleaned = clean_text(text)
        X_tfidf = vectorizer.transform([cleaned])
        # Return the predicted label from the classifier
        return clf.predict(X_tfidf)[0]
    
    return predict_sentiment

# Main block: Train the model if the script is run directly
if __name__ == "__main__":
    train_sentiment_model(csv_path="sample_data.csv")
