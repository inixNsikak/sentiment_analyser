# Import all necessarry packages
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import sklearn
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# *from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import warnings
warnings.simplefilter("ignore")

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')


# Initialize the Tkinter window
root = tk.Tk()
root.title("Sentiment Analyser")
root.geometry('400x400')

# We initialize y_test and y_pred
# y_test and y_pred will be used later in the project
y_test=""
y_pred=""

# Load your dataset into a DataFrame (replace 'your_dataset.csv' with your dataset)
data = None

def load_dataset():
    global data
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        # info_label.config(text=f"Loaded dataset: {file_path}")
        info_label.config(text=f"Data Loaded!")
        preprocess_button.config(state="normal")
        # sentiment_button.config(state="normal")

# Fucton to Preprocess the text 
def preprocess_text(text):
    
    # # List of columns to extract
    # columns_to_extract = ['text']
    # # Code to Extract the desired columns and store in a new DataFrame
    # text_tweets_df = tweets_df[columns_to_extract]

    # Convert to string and remove special characters, URLs, and usernames
    tweet = str(text)
    tweet = re.sub(r'http\S+|www\S+|@[^\s]+', '', tweet)
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)

    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenizations
    tokens = word_tokenize(tweet)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a string
    preprocessed_tweet = ' '.join(tokens)
        
    return preprocessed_tweet
    

def preprocess_data():
    if data is not None:
        data['preprocessed_text'] = data['text'].apply(preprocess_text)
        print(data['preprocessed_text'])
        info_label.config(text="Data preprocessed!")
        find_sentiment_button.config(state="normal")

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X = None

# Sentiment Analysis
# Initialize the SentimentIntensityAnalyzer for sentiment evaluation
sia = SentimentIntensityAnalyzer()
        
def find_sentiment(post):
    try:
        compound_score = sia.polarity_scores(post)["compound"]
        if compound_score > 0:
            return "Positive"
        elif compound_score < 0:
            return "Negative"
        else:
            return "Neutral"
    except:
        return "Neutral"

    return sentiment

def check_sentiment():
    if data is not None:
        data['sentiment'] = data['text'].apply(find_sentiment)
        print(data['sentiment'])
        info_label.config(text="Sentiment Gotten!")
        train_and_evaluate_model_button.config(state="normal")   

def train_and_evaluate_model(df):
    # Split the dataset into training and testing sets
    X = df['preprocessed_text']
    y = df['sentiment']

    # Convert text data to numerical data using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    global y_test, y_pred

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes model
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = naive_bayes.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # Print classification report
    print('Classification Report:')
    print(classification_report(y_test, y_pred))




def train_and_evaluate():
    if data is not None:
        # data['sentiment'] = data['text'].apply(find_sentiment)
        train_and_evaluate_model(data)
        info_label.config(text="All done!")
        # result_label.config(text=f"Accuracy:  {accuracy}")
        # compound_label.config(text=f"Compound Sentiment Score (VADER): {sentiment_scores['compound']}")
        classification_report_button.config(state="normal")


# Show the classification report and a graph
def show_results():
    global y_test, y_pred
    # # Make predictions on the test set
    # y_pred = naive_bayes.predict(X_test)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display results in a window
    result_window = tk.Toplevel(root)
    result_window.title("Classification Report and Graph")

    # Display accuracy
    accuracy_label = tk.Label(result_window, text=f"Accuracy: {accuracy}")
    accuracy_label.pack()

    # Display classification report
    report_label = tk.Label(result_window, text="Classification Report:")
    report_label.pack()
    report_text = tk.Text(result_window, height=20, width=60)
    report_text.insert(tk.END, classification_report(y_test, y_pred))
    report_text.pack()

    # Display confusion matrix as a heatmap
    conf_matrix_label = tk.Label(result_window, text="Confusion Matrix:")
    conf_matrix_label.pack()
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the plot as an image
    plt.close()

    # Display the saved confusion matrix image
    img = tk.PhotoImage(file='confusion_matrix.png')
    canvas = tk.Canvas(result_window, width=img.width(), height=img.height())
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.pack()

    # Create a bar plot for sentiment distribution in the test set
    sentiment_counts = y_test.value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution in Test Set')
    plt.savefig('sentiment_distribution.png')  # Save the plot as an image
    plt.close()

    # Display the saved sentiment distribution image
    img_sentiment = tk.PhotoImage(file='sentiment_distribution.png')
    canvas_sentiment = tk.Canvas(result_window, width=img_sentiment.width(), height=img_sentiment.height())
    canvas_sentiment.create_image(0, 0, anchor=tk.NW, image=img_sentiment)
    canvas_sentiment.pack()

    report_label.pack()
    report_text.pack()
    graph_label.pack()
    canvas.pack()

        # Update the graph in the canvas
        # You will need to implement your own code to display a graph in the canvas

# GUI components
load_button = tk.Button(root, text="Load Dataset", command=load_dataset)
preprocess_button = tk.Button(root, text="Preprocess Data", state="disabled", command=preprocess_data)
find_sentiment_button = tk.Button(root, text="Find Sentiment", state="disabled", command=check_sentiment)
train_and_evaluate_model_button = tk.Button(root, text="Train & Evaluate", state="disabled", command=train_and_evaluate)
info_label = tk.Label(root, text="")
result_label = tk.Label(root, text="")
# graph_label= tk.Label(root,text)pack()
# compound_label = tk.Label(root, text="")
classification_report_button = tk.Button(root, text="Show Classification Report", state="disabled", command=show_results)
# sentiment_button = tk.Button(root, text="Sentiment Analysis Button")  # Define sentiment_button

# Layout
load_button.pack()
preprocess_button.pack()
find_sentiment_button.pack()

train_and_evaluate_model_button.pack()
info_label.pack()
result_label.pack()
# graph_label.pack()
# compound_label.pack()
classification_report_button.pack()
# sentiment_button.pack()  # Pack the sentiment_button

root.mainloop()
