# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:03:15 2023
python = 3.9
@author: User
"""

#Topic modeling using LDA
import pandas as pd
import gensim
from gensim import corpora

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pyLDAvis.gensim_models
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
import pyLDAvis

# Function to load and preprocess dataset
def load_and_preprocess(file_path, raw_col_name, clean_col_name, sentiment):
    df = pd.read_csv(file_path)
    df['sentiment'] = sentiment
    return df.rename(columns={raw_col_name: 'raw_review', clean_col_name: 'clean_review'})

# Load the datasets
positive_reviews_test = load_and_preprocess("positive_reviews_test.csv", 'Positive_Review', 'positive_reviews_clean', 1)
positive_reviews_test = positive_reviews_test.sample(n=20,random_state=42)
negative_reviews_test = load_and_preprocess("negative_reviews_test.csv", 'Negative_Review', 'negative_reviews_clean', 0)
negative_reviews_test = negative_reviews_test.sample(n=20,random_state=42)
all_reviews = pd.concat([positive_reviews_test, negative_reviews_test]) #.sample(frac=1, random_state=42).reset_index(drop=True)

# Shuffle the dataset
all_reviews = all_reviews.sample(frac=1).reset_index(drop=True)

# Convert reviews to string and remove null values
all_reviews['raw_review'] = all_reviews['raw_review'].astype(str)
all_reviews['clean_review'] = all_reviews['clean_review'].astype(str)
all_reviews = all_reviews.dropna(subset=['raw_review', 'clean_review', 'sentiment'])




# Initialize tools
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

def process_text(text):
    # Clean the text
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)
    
    # Remove stop words
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # Stem the tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    return stemmed_tokens

def top_words_per_topic(lda_model, n):
    top_words = {}
    for i in range(lda_model.num_topics):
        words = lda_model.show_topic(i, topn=n)
        top_words[i] = [word for word, _ in words]
    return top_words

# Function to process and apply LDA
def apply_lda(reviews, review_type):
    processed_reviews = reviews.map(process_text)
    
    # Construct a document-term matrix
    dictionary = corpora.Dictionary(processed_reviews)
    corpus = [dictionary.doc2bow(text) for text in processed_reviews]
    
    # Generate the LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
    
    # Extract topics and their top words
    top_n_words = top_words_per_topic(ldamodel, 10)
    topic_words = pd.DataFrame({
        'review_type': review_type,
        'topic_id': list(top_n_words.keys()),
        'top_words': list(top_n_words.values())
    })
    
    # Prepare visualization
    lda_vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
    
    return ldamodel, lda_vis, topic_words

# Initialize placeholders for all outputs
all_topic_words = pd.DataFrame()

# Separate positive and negative reviews
positive_reviews = all_reviews[all_reviews['sentiment'] == 1]
negative_reviews = all_reviews[all_reviews['sentiment'] == 0]

# Loop through raw and clean reviews
for review_column, review_type in [('raw_review', 'Raw'), ('clean_review', 'Clean')]:
    for reviews, sentiment_label in [(positive_reviews, 'Positive'), (negative_reviews, 'Negative')]:
        ldamodel, lda_vis, topic_words = apply_lda(reviews[review_column], f"{sentiment_label}_{review_type}")
        
        # Save visualization
        pyLDAvis.save_html(lda_vis, f"{sentiment_label}_{review_type}_reviews.html")
        
        # Append topic words
        all_topic_words = pd.concat([all_topic_words, topic_words], ignore_index=True)

# Save topics
all_topic_words.to_csv('LDA_topic_words_comparison_sample.csv', index=False)

# Output
print("LDA modeling and visualization completed for both raw and clean reviews.")
