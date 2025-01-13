# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 00:35:03 2023

@author: User
"""
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Updated import
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


st = time.time()

# Function to load and preprocess dataset
def load_and_preprocess(file_path, raw_col_name, clean_col_name, sentiment):
    df = pd.read_csv(file_path)
    df['sentiment'] = sentiment
    return df.rename(columns={raw_col_name: 'raw_review', clean_col_name: 'clean_review'})

# Load and preprocess the training dataset
positive_reviews_train = load_and_preprocess("positive_reviews_train.csv",'Positive_Review','positive_reviews_clean', 1)
negative_reviews_train = load_and_preprocess("negative_reviews_train.csv",'Negative_Review','negative_reviews_clean', 0)
reviews_dataset_train = pd.concat([positive_reviews_train, negative_reviews_train]).sample(frac=1, random_state=42).reset_index(drop=True)

# Extract features and labels for training
X_raw = reviews_dataset_train['raw_review']
X_clean = reviews_dataset_train['clean_review']
y = reviews_dataset_train['sentiment']

# Tokenize and pad raw reviews for training
tokenizer_raw = Tokenizer()
tokenizer_raw.fit_on_texts(X_raw)
X_raw_seq = tokenizer_raw.texts_to_sequences(X_raw)
X_raw_padded = pad_sequences(X_raw_seq, maxlen=100)

# Tokenize and pad clean reviews for training
tokenizer_clean = Tokenizer()
tokenizer_clean.fit_on_texts(X_clean)
X_clean_seq = tokenizer_clean.texts_to_sequences(X_clean)
X_clean_padded = pad_sequences(X_clean_seq, maxlen=100)

# Initialize the StratifiedKFold object for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize global variables for the models
model_raw = None
model_clean = None

# Function to train and evaluate the model
def train_and_evaluate(X, tokenizer, model_name, is_clean=False):
    global model_raw, model_clean  # Use global variables for models

    fold_no = 1
    for train, val in skf.split(X, y):
        # Define the LSTM model architecture
        model = Sequential()
        model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=100))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        print(f'Training {model_name} model for fold {fold_no}...')
        model.fit(X[train], y.iloc[train], epochs=5, batch_size=32, verbose=1)

        fold_no += 1

        # Update the global model variables with the last fold's model
        if fold_no > n_splits:
            if is_clean:
                model_clean = model
            else:
                model_raw = model

# Train models on raw and clean reviews
train_and_evaluate(X_raw_padded, tokenizer_raw, "raw", is_clean=False)
train_and_evaluate(X_clean_padded, tokenizer_clean, "clean", is_clean=True)

# Load and preprocess the test dataset (function is similar to the one used for training data)
# Ensure you have 'positive_reviews_test.csv' and 'negative_reviews_test.csv' ready
positive_reviews_test = load_and_preprocess("positive_reviews_test.csv",'Positive_Review','positive_reviews_clean', 1)
negative_reviews_test = load_and_preprocess("negative_reviews_test.csv", 'Negative_Review','negative_reviews_clean', 0)
reviews_dataset_test = pd.concat([positive_reviews_test, negative_reviews_test]).sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare raw and clean test sets
X_raw_test = reviews_dataset_test['raw_review']
X_clean_test = reviews_dataset_test['clean_review']
y_test = reviews_dataset_test['sentiment']

# Tokenize and pad the raw and clean test reviews
X_raw_test_seq = tokenizer_raw.texts_to_sequences(X_raw_test)
X_raw_test_padded = pad_sequences(X_raw_test_seq, maxlen=100)

X_clean_test_seq = tokenizer_clean.texts_to_sequences(X_clean_test)
X_clean_test_padded = pad_sequences(X_clean_test_seq, maxlen=100)

# Function to evaluate the model on the test set
def evaluate_on_test_set(X_test_padded, y_test, model, model_name):
    test_scores = model.evaluate(X_test_padded, y_test, verbose=1)
    print(f'\nTest Loss for {model_name} reviews: {test_scores[0]}')
    print(f'Test Accuracy for {model_name} reviews: {test_scores[1]*100}%')
    
    # Calculate precision, recall, and F1 score on the test set
    test_predictions = (model.predict(X_test_padded) > 0.5).astype(int)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    
    print(f'Test Precision for {model_name} reviews: {test_precision}')
    print(f'Test Recall for {model_name} reviews: {test_recall}')
    print(f'Test F1 for {model_name} reviews: {test_f1}')
    print('------------------------------------------------------------------------\n')

# Evaluate models on the test set
evaluate_on_test_set(X_raw_test_padded, y_test, model_raw, "raw")
evaluate_on_test_set(X_clean_test_padded, y_test, model_clean, "clean")

print(f'Total runtime: {time.time() - st} seconds')

#%%

# After the evaluate_on_test_set function calls
print(f'Total runtime: {time.time() - st} seconds')

# Save the models
if model_raw is not None:
    model_raw.save('LSTM_model_raw.h5')
    print('Raw model saved as model_raw.h5')

if model_clean is not None:
    model_clean.save('LSTM_model_clean.h5')
    print('Clean model saved as model_clean.h5')

