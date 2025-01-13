# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 00:35:03 2023

@author: User
"""

import pandas as pd
import re
from langdetect import detect
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions
import matplotlib.pyplot as plt
import seaborn as sns

with open("Hotel_Reviews.csv") as f:
    hotel_reviews = pd.read_csv('Hotel_Reviews.csv')


# Split into negative and positive datasets
negative_reviews = hotel_reviews.copy()
negative_reviews = negative_reviews[['Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date',
       'Average_Score', 'Hotel_Name', 'Reviewer_Nationality',
       'Negative_Review', 'Review_Total_Negative_Word_Counts', 'Tags']]

positive_reviews = hotel_reviews.copy()
positive_reviews = positive_reviews[['Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date',
       'Average_Score', 'Hotel_Name', 'Reviewer_Nationality',
       'Positive_Review', 'Review_Total_Positive_Word_Counts', 'Tags']]


# Data cleaning


contractions_dictionary = {
    "I'm": "I am",
    "I'm'a": "I am about to",
    "I'm'o": "I am going to",
    "I've": "I have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'd": "I would",
    "I'd've": "I would have",
    "Whatcha": "What are you",
    "amn't": "am not",
    "ain't": "are not",
    "aren't": "are not",
    "'cause": "because",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "daren't": "dare not",
    "daresn't": "dare not",
    "dasn't": "dare not",
    "didn't": "did not",
    "didn’t": "did not",
    "don't": "do not",
    "don’t": "do not",
    "doesn't": "does not",
    "e'er": "ever",
    "everyone's": "everyone is",
    "finna": "fixing to",
    "gimme": "give me",
    "gon't": "go not",
    "gonna": "going to",
    "gotta": "got to",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he've": "he have",
    "he's": "he is",
    "he'll": "he will",
    "he'll've": "he will have",
    "he'd": "he would",
    "he'd've": "he would have",
    "here's": "here is",
    "how're": "how are",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how's": "how is",
    "how'll": "how will",
    "isn't": "is not",
    "it's": "it is",
    "'tis": "it is",
    "'twas": "it was",
    "it'll": "it will",
    "it'll've": "it will have",
    "it'd": "it would",
    "it'd've": "it would have",
    "kinda": "kind of",
    "let's": "let us",
    "luv": "love",
    "ma'am": "madam",
    "may've": "may have",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "ne'er": "never",
    "o'": "of",
    "o'clock": "of the clock",
    "ol'": "old",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "o'er": "over",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shalln't": "shall not",
    "shan't've": "shall not have",
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    "she'd've": "she would have",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "somebody's": "somebody is",
    "someone's": "someone is",
    "something's": "something is",
    "sux": "sucks",
    "that're": "that are",
    "that's": "that is",
    "that'll": "that will",
    "that'd": "that would",
    "that'd've": "that would have",
    "'em": "them",
    "there're": "there are",
    "there's": "there is",
    "there'll": "there will",
    "there'd": "there would",
    "there'd've": "there would have",
    "these're": "these are",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they'd": "they would",
    "they'd've": "they would have",
    "this's": "this is",
    "this'll": "this will",
    "this'd": "this would",
    "those're": "those are",
    "to've": "to have",
    "wanna": "want to",
    "wasn't": "was not",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we'd": "we would",
    "we'd've": "we would have",
    "weren't": "were not",
    "what're": "what are",
    "what'd": "what did",
    "what've": "what have",
    "what's": "what is",
    "what'll": "what will",
    "what'll've": "what will have",
    "when've": "when have",
    "when's": "when is",
    "where're": "where are",
    "where'd": "where did",
    "where've": "where have",
    "where's": "where is",
    "which's": "which is",
    "who're": "who are",
    "who've": "who have",
    "who's": "who is",
    "who'll": "who will",
    "who'll've": "who will have",
    "who'd": "who would",
    "who'd've": "who would have",
    "why're": "why are",
    "why'd": "why did",
    "why've": "why have",
    "why's": "why is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "you're": "you are",
    "you've": "you have",
    "you'll've": "you shall have",
    "you'll": "you will",
    "you'd": "you would",
    "you'd've": "you would have",

}

remove_words = [
    'nothing', 'none', 'nil', 'n a', 'all good', "cannot complain", 
    "not complain", "not bad", "no issues", "no problem",
    "nothing to complain about", "nothing special", "average",
    "mediocre", "unremarkable", "no comment", "not much",
    "no major issues", "no significant problems", "fine",
    "acceptable", "satisfactory", "ok", "nothing remarkable",
    "not a big deal", "no big issues", "nothing noteworthy",
    "not too bad", "could be worse", "nothing all great"
]

# Adjustments for negations handled with "NOT_"
# For example, if "not bad" is a phrase you're looking for, it might become "NOT_bad" after handling negations
adjusted_negations = ["NOT_" + word if 'not ' in word else word for word in remove_words]

# The final list might look something like this, combining both expansions and negation adjustments
final_remove_words = remove_words + adjusted_negations

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


idioms = {
    "a room fit for a king": "a very luxurious room",
    "a home away from home": "a comfortable room",
    "cramped like sardines": "packed very tightly together",
    "bent over backwards": "made a great effort",
    "go the extra mile": "do more than what is required or expected",
    "pull out all the stops": "do everything possible",
    "melt in your mouth": "extremely tender and flavorful",
    "a feast for the eyes": "a very beautiful sight",
    "good enough to eat": "looks very appealing",
    "hit the spot": "satisfying",
    "a mixed bag": "a collection of",
    "over the moon": "extremely happy or pleased",
    "spick and span": "very clean and tidy",
    "squeaky clean": "extremely clean",
    "a pigsty": "very dirty",
    "bang for your buck": "value for money",
    "an arm and a leg": "very high price",
    "a steal": "an extremely good deal ",
    "in the heart of": "in the center of",
    "off the beaten path": "not commonly visited",
    "a stone's throw away": "a very short distance away"
}


def create_space_contractions(contractions_dict):
    new_dict = {key.replace("'", " "): value for key, value in contractions_dict.items()}
    return new_dict


new_contractions = create_space_contractions(contractions_dictionary)

for contraction, expansion in new_contractions.items():
    contractions.add(contraction, expansion)


def clean_text(text_series, remove_words, idioms):
    '''Cleans text series by handling negations, expanding contractions, handling idioms, and removing unwanted words.'''

    def expand_contractions(text):
        return contractions.fix(text)

    def handle_negations(text):
        negation_words = ['not', 'no', 'never', 'none']
        tokens = word_tokenize(text)
        negated_text = []
        negation_flag = False
        for token in tokens:
            if token in negation_words:
                negation_flag = True
                continue
            if negation_flag:
                negated_text.append(f"NOT_{token}")
                negation_flag = False
            else:
                negated_text.append(token)
        return ' '.join(negated_text)

    def handle_idioms(text, idioms):
        for idiom, replacement in idioms.items():
            text = text.replace(idiom, replacement)
        return text

    # Start the cleaning process
    # Convert to lowercase
    text_series = text_series.str.lower()

    # Expand contractions
    text_series = text_series.apply(expand_contractions)

    # Handle negations
    text_series = text_series.apply(handle_negations)

    # Handle idioms
    text_series = text_series.apply(lambda x: handle_idioms(x, idioms))

    # Remove non-ASCII characters
    text_series = text_series.str.replace(r'[^\x00-\x7F]+', '', regex=True)

    # Remove stop words
    stop_words = set(STOP_WORDS)
    text_series = text_series.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Remove words from the specified list
    remove_words_set = set(remove_words)
    text_series = text_series.apply(lambda x: ' '.join([word for word in x.split() if word not in remove_words_set]))

    # Lemmatize the text
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text_series = text_series.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    return text_series


def is_empty_or_whitespace(text):
    return text is None or text.strip() == ''


print("Filtering out 'No Negative' and 'No Positive' reviews...")
negative_reviews_filter_1 = negative_reviews.loc[negative_reviews['Negative_Review'] != 'No Negative'].copy()
positive_reviews_filter_1 = positive_reviews.loc[positive_reviews['Positive_Review'] != 'No Positive'].copy()

print("Recalculating word count for each review...")
negative_reviews_filter_1.loc[:, 'Negative_review_word_count'] = negative_reviews_filter_1['Negative_Review'].str.split().str.len()
positive_reviews_filter_1.loc[:, 'Positive_review_word_count'] = positive_reviews_filter_1['Positive_Review'].str.split().str.len()

print("Applying clean_text function to reviews...")
negative_reviews_filter_1.loc[:, 'negative_reviews_clean'] = clean_text(negative_reviews_filter_1['Negative_Review'], final_remove_words, idioms)
positive_reviews_filter_1.loc[:, 'positive_reviews_clean'] = clean_text(positive_reviews_filter_1['Positive_Review'], final_remove_words, idioms)

print("Filtering out empty or whitespace-only reviews...")
negative_reviews_clean = negative_reviews_filter_1.loc[~negative_reviews_filter_1['negative_reviews_clean'].apply(is_empty_or_whitespace)].copy()
positive_reviews_clean = positive_reviews_filter_1.loc[~positive_reviews_filter_1['positive_reviews_clean'].apply(is_empty_or_whitespace)].copy()

print("Recalculating word count for cleaned reviews...")
negative_reviews_clean.loc[:, 'negative_reviews_word_count_clean'] = negative_reviews_clean['negative_reviews_clean'].str.split().str.len()
positive_reviews_clean.loc[:, 'positive_reviews_word_count_clean'] = positive_reviews_clean['positive_reviews_clean'].str.split().str.len()

print("Filtering cleaned reviews with less than 2 words...")
negative_reviews_clean = negative_reviews_clean.loc[negative_reviews_clean['negative_reviews_word_count_clean'] > 2].copy()
positive_reviews_clean = positive_reviews_clean.loc[positive_reviews_clean['positive_reviews_word_count_clean'] > 2].copy()




#%%

# Visualization of word counts

import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogram of negative review word count
plt.figure(figsize=(10, 6))
sns.histplot(negative_reviews_filter_1['Negative_review_word_count'], bins=50, kde=True)
plt.title('Distribution of negative review word counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()


# Plot histogram of positive review conut
plt.figure(figsize=(10, 6))
sns.histplot(positive_reviews_filter_1['Positive_review_word_count'], bins=50, kde=True)
plt.title('Distribution of positive review word counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()


# Plot both distirbutions toghether

import seaborn as sns
import matplotlib.pyplot as plt

# Set the size of the figure
plt.figure(figsize=(12, 6))

# Plot the distribution of negative review word counts
sns.histplot(negative_reviews_filter_1['Negative_review_word_count'], bins=30, color="red", label='Negative Reviews', kde=True, alpha=0.5)

# Plot the distribution of positive review word counts
sns.histplot(positive_reviews_filter_1['Positive_review_word_count'], bins=30, color="blue", label='Positive Reviews', kde=True, alpha=0.5)

# Add legend and titles
plt.legend()
plt.title('Comparison of Word Counts in Positive and Negative Reviews')
plt.xlabel('Word Count')
plt.ylabel('Frequency')

# Show the plot
plt.show()

quartiles_negative_word_counts = negative_reviews_filter_1['Negative_review_word_count'].describe(percentiles=[0.25, 0.5, 0.75])
quartiles_positive_word_counts = positive_reviews_filter_1['Positive_review_word_count'].describe(percentiles=[0.25, 0.5, 0.75])



#%%

# Text analysis using TextBlob

from textblob import TextBlob
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    testimonial = TextBlob(text)
    polarity = testimonial.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'


# Analysis on raw data
print("Analyzing sentiment on raw review data...")
# Analyze sentiment for negative reviews
# Convert pandas DataFrame to Dask DataFrame
dask_negative_reviews_clean = dd.from_pandas(negative_reviews_clean, npartitions=4)

# Apply the function to each partition and compute with progress bar for negative reviews
with ProgressBar():
    negative_sentiments = dask_negative_reviews_clean.map_partitions(lambda df: df['Negative_Review'].apply(analyze_sentiment_textblob), meta=('sentiment', 'str')).compute()

# Assign the computed sentiments back to the original DataFrame for negative reviews
negative_reviews_clean['sentiment_textblob_raw'] = negative_sentiments

# Analyze sentiment for positive reviews
# Convert pandas DataFrame to Dask DataFrame
dask_positive_reviews_clean = dd.from_pandas(positive_reviews_clean, npartitions=4)

# Apply the function to each partition and compute with progress bar for positive reviews
with ProgressBar():
    positive_sentiments = dask_positive_reviews_clean.map_partitions(lambda df: df['Positive_Review'].apply(analyze_sentiment_textblob), meta=('sentiment', 'str')).compute()

# Assign the computed sentiments back to the original DataFrame for positive reviews
positive_reviews_clean['sentiment_textblob_raw'] = positive_sentiments


# Analysis on clean data
print("Analyzing sentiment on cleaned review data...")
# Analyze sentiment for negative reviews
# Convert pandas DataFrame to Dask DataFrame
dask_negative_reviews_clean = dd.from_pandas(negative_reviews_clean, npartitions=4)

# Apply the function to each partition and compute with progress bar for negative reviews
with ProgressBar():
    negative_sentiments = dask_negative_reviews_clean.map_partitions(lambda df: df['negative_reviews_clean'].apply(analyze_sentiment_textblob), meta=('sentiment', 'str')).compute()

# Assign the computed sentiments back to the original DataFrame for negative reviews
negative_reviews_clean['sentiment_textblob_clean'] = negative_sentiments

# Analyze sentiment for positive reviews
# Convert pandas DataFrame to Dask DataFrame
dask_positive_reviews_clean = dd.from_pandas(positive_reviews_clean, npartitions=4)

# Apply the function to each partition and compute with progress bar for positive reviews
with ProgressBar():
    positive_sentiments = dask_positive_reviews_clean.map_partitions(lambda df: df['positive_reviews_clean'].apply(analyze_sentiment_textblob), meta=('sentiment', 'str')).compute()

# Assign the computed sentiments back to the original DataFrame for positive reviews
positive_reviews_clean['sentiment_textblob_clean'] = positive_sentiments

#%%

# Match sentiment in raw and cleaned datasets

negative_reviews_clean['sentiment_match'] = negative_reviews_clean['sentiment_textblob_raw'] == negative_reviews_clean['sentiment_textblob_clean']
positive_reviews_clean['sentiment_match'] = positive_reviews_clean['sentiment_textblob_raw'] == positive_reviews_clean['sentiment_textblob_clean']

#%%

cond_neg = (negative_reviews_clean['sentiment_match']==True) & (negative_reviews_clean['sentiment_textblob_raw']=='Negative')
negative_reviews_matched = negative_reviews_clean[cond_neg]

cond_pos = (positive_reviews_clean['sentiment_match']==True) & (positive_reviews_clean['sentiment_textblob_raw']=='Positive')
positive_reviews_matched = positive_reviews_clean[cond_pos]

#%%

# Equalizing samples
sample_size = negative_reviews_matched.shape[0]

negative_reviews_final = negative_reviews_matched.copy()
positive_reviews_final = positive_reviews_matched.sample(sample_size)

#%%

# Extracting test set


# Define sample size
test_sample_size = 6000  

# Sample the raw datasets
negative_reviews_test = negative_reviews_final.sample(n=test_sample_size, random_state=42)
positive_reviews_test = positive_reviews_final.sample(n=test_sample_size, random_state=42)


# Drop the sampled indices from the datasets to create training sets
negative_reviews_train = negative_reviews_final.drop(negative_reviews_test.index)
positive_reviews_train = positive_reviews_final.drop(positive_reviews_test.index)




#%%

plt.figure(figsize=(10, 6))
sns.histplot(positive_reviews_train['Positive_review_word_count'], bins=50, kde=True)
plt.title('Distribution of Word Counts in Positive train dataset')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()



#%%


# Export the cleaned datasets
negative_reviews_final.to_csv("negative_reviews_all.csv")
positive_reviews_final.to_csv("positive_reviews_all.csv")

negative_reviews_test.to_csv("negative_reviews_test.csv")
positive_reviews_test.to_csv("positive_reviews_test.csv")

negative_reviews_train.to_csv("negative_reviews_train.csv")
positive_reviews_train.to_csv("positive_reviews_train.csv")



#%%
negative_reviews_raw_final.to_csv("negative_reviews_raw_train.csv")
positive_reviews_raw_final.to_csv("positive_reviews_raw_train.csv")

negative_reviews_clean_final.to_csv("negative_reviews_clean_train.csv")
positive_reviews_clean_final.to_csv("positive_reviews_clean_train.csv")

#%%

# Visualize response distributions

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
with open("negative_reviews_raw_train.csv") as f:
    negative_reviews_raw_final = pd.read_csv(f)

with open("positive_reviews_raw_train.csv") as f:
    positive_reviews_raw_final = pd.read_csv(f)
    
with open("negative_reviews_clean_train.csv") as f:
    negative_reviews_clean_final = pd.read_csv(f)

with open("positive_reviews_clean_train.csv") as f:
    positive_reviews_clean_final = pd.read_csv(f)

# Define a list of tuples, each containing a dataframe and the relevant word count column name
datasets = [
    (negative_reviews_raw_final, 'Negative_review_word_count', 'Negative Raw Reviews'),
    (positive_reviews_raw_final, 'Positive_review_word_count', 'Positive Raw Reviews'),
    (negative_reviews_clean_final, 'Negative_review_word_count', 'Negative Clean Reviews'),
    (positive_reviews_clean_final, 'Positive_review_word_count', 'Positive Clean Reviews')
]

# Loop over the datasets and plot the histograms
for df, word_count_column, title in datasets:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[word_count_column], bins=50, kde=True)
    plt.title(f'Distribution of Word Counts in {title}')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.show()





