# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:16:38 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Script using BERT for sentiment analysis with attention heatmaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Function to load and preprocess dataset
def load_and_preprocess(file_path, raw_col_name, clean_col_name, sentiment):
    df = pd.read_csv(file_path)
    df['sentiment'] = sentiment
    return df.rename(columns={raw_col_name: 'raw_review', clean_col_name: 'clean_review'})

# Load the datasets
positive_reviews_test = load_and_preprocess("positive_reviews_test.csv", 'Positive_Review', 'positive_reviews_clean', 1)
negative_reviews_test = load_and_preprocess("negative_reviews_test.csv", 'Negative_Review', 'negative_reviews_clean', 0)
all_reviews = pd.concat([positive_reviews_test, negative_reviews_test]).sample(frac=1, random_state=42).reset_index(drop=True)

# Shuffle the dataset
all_reviews = all_reviews.sample(frac=1).reset_index(drop=True)

# Convert reviews to string and remove null values
all_reviews['raw_review'] = all_reviews['raw_review'].astype(str)
all_reviews['clean_review'] = all_reviews['clean_review'].astype(str)
all_reviews = all_reviews.dropna(subset=['raw_review', 'clean_review', 'sentiment'])

# Load BERT model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)

# Function to predict sentiment and extract attention
def predict_with_attention(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        logits = outputs.logits
        attentions = outputs.attentions  # Attention weights from all layers

        # Predicted sentiment
        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_sentiment = 1 if predicted_label == 1 else 0

        return predicted_sentiment, attentions, inputs['input_ids']
    except Exception as e:
        print(f"Error processing text: {text}")
        print(f"Error message: {str(e)}")
        return None, None, None

# Function to visualize attention heatmaps
def visualize_attention(attentions, input_ids, layer=-1, head=0):
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    attention_weights = attentions[layer][0][head].detach().numpy()  # Shape: (seq_len, seq_len)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap="viridis", square=True)
    plt.title(f"Attention Heatmap (Layer {layer+1}, Head {head+1})")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.show()

# Predict sentiments and generate heatmaps for a few examples
example_reviews = all_reviews['raw_review'].head(10)  # Change to any subset of reviews
for i, review in enumerate(example_reviews):
    predicted_sentiment, attentions, input_ids = predict_with_attention(review)
    if attentions is not None:
        print(f"Review {i+1}: {review}")
        print(f"Predicted Sentiment: {'Positive' if predicted_sentiment == 1 else 'Negative'}")
        visualize_attention(attentions, input_ids, layer=-1, head=0)  # Last layer, first head

# Evaluate model performance
def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}


#%%
# Predict for all reviews
all_reviews['predicted_sentiment'] = all_reviews['raw_review'].apply(lambda x: predict_with_attention(x)[0])

# Remove rows with None predictions
all_reviews = all_reviews.dropna(subset=['predicted_sentiment'])

# Evaluate predictions
performance = evaluate_performance(all_reviews['sentiment'], all_reviews['predicted_sentiment'])
print("\nModel Performance:")
for metric, value in performance.items():
    print(f"{metric}: {value:.4f}")

# Save results
all_reviews.to_csv('test_reviews_with_attention_BERT.csv', index=False)
