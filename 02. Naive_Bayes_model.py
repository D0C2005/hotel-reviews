# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 00:35:03 2023

@author: User
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st=time.time()

with open("positive_reviews_test.csv") as f:
    positive_reviews_test = pd.read_csv(f)

with open("negative_reviews_test.csv") as f:
    negative_reviews_test = pd.read_csv(f)


with open("positive_reviews_train.csv") as f:
    positive_reviews_train = pd.read_csv(f)

with open("negative_reviews_train.csv") as f:
    negative_reviews_train = pd.read_csv(f)



# 'review' column contains the review text
# 'sentiment' column contains the sentiment (0 for negative, 1 for positive)


# Dataset preparation
positive_reviews_train['sentiment'] = 1
negative_reviews_train['sentiment'] = 0


positive_reviews_train.rename(columns={'Positive_Review': 'raw_review', 'positive_reviews_clean': 'clean_review'}, inplace=True)
negative_reviews_train.rename(columns={'Negative_Review': 'raw_review', 'negative_reviews_clean': 'clean_review'}, inplace=True)


positive_reviews_train = positive_reviews_train[['raw_review','clean_review', 'sentiment']]
negative_reviews_train = negative_reviews_train[['raw_review','clean_review', 'sentiment']]


# Combine the datasets
reviews_dataset_train = pd.concat([positive_reviews_train, negative_reviews_train]).sample(frac=1, random_state=42).reset_index(drop=True)


# Prepare datasets
X_raw = reviews_dataset_train['raw_review']
X_clean = reviews_dataset_train['clean_review']
y = reviews_dataset_train['sentiment']

# Define the number of folds for k-fold cross-validation
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Create a pipeline that first converts text data into token counts (bag-of-words) and then applies Naive Bayes Classifier
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())


# Perform k-fold cross-validation for raw text reviews
scores_raw = cross_val_score(pipeline, X_raw, y, cv=kf, scoring='accuracy')
print(f"Raw text reviews - Accuracy across {n_folds} folds: {scores_raw}")
print(f"Raw text reviews - Mean Accuracy: {np.mean(scores_raw)}")

# Perform k-fold cross-validation for clean text reviews
scores_clean = cross_val_score(pipeline, X_clean, y, cv=kf, scoring='accuracy')
print(f"Clean text reviews - Accuracy across {n_folds} folds: {scores_clean}")
print(f"Clean text reviews - Mean Accuracy: {np.mean(scores_clean)}")




# Dataset preparation
positive_reviews_test['sentiment'] = 1
negative_reviews_test['sentiment'] = 0


positive_reviews_test.rename(columns={'Positive_Review': 'raw_review', 'positive_reviews_clean': 'clean_review'}, inplace=True)
negative_reviews_test.rename(columns={'Negative_Review': 'raw_review', 'negative_reviews_clean': 'clean_review'}, inplace=True)


positive_reviews_test = positive_reviews_test[['raw_review','clean_review', 'sentiment']]
negative_reviews_test = negative_reviews_test[['raw_review','clean_review', 'sentiment']]


# Combine the datasets
reviews_dataset_test = pd.concat([positive_reviews_test, negative_reviews_test]).sample(frac=1, random_state=42).reset_index(drop=True)


# Prepare datasets
X_raw_test = reviews_dataset_test['raw_review']
X_clean_test = reviews_dataset_test['clean_review']
y_test = reviews_dataset_test['sentiment']


# Train the model on the entire training set for raw reviews
pipeline_raw = make_pipeline(CountVectorizer(), MultinomialNB())
pipeline_raw.fit(X_raw, y)

# Train the model on the entire training set for clean reviews
pipeline_clean = make_pipeline(CountVectorizer(), MultinomialNB())
pipeline_clean.fit(X_clean, y)

# Predict on the raw review test set
predictions_raw_test = pipeline_raw.predict(X_raw_test)

# Predict on the clean review test set
predictions_clean_test = pipeline_clean.predict(X_clean_test)

reviews_dataset_test['predictions_raw']=predictions_raw_test
reviews_dataset_test['predictions_clean']=predictions_clean_test
reviews_dataset_test.to_csv('results_Naive_bayes_test_set.csv',index=False)

#%%
# Evaluate the performance
print("Performance on Raw Review Test Set:")
print(f"Accuracy: {accuracy_score(y_test, predictions_raw_test):.8f}")
print(f"Precision:{precision_score(y_test, predictions_raw_test, average='weighted'):.8f}")
print(f"Recall: {recall_score(y_test, predictions_raw_test, average='weighted'):.8f}")
print(f"F1 score: {f1_score(y_test, predictions_raw_test, average='weighted'):.8f}")
print(f"\nClassification Report:\n {classification_report(y_test, predictions_raw_test)}")
print(f"\nConfusion Matrix:\n {confusion_matrix(y_test, predictions_raw_test)}")


print("\nPerformance on Clean Review Test Set:")
print(f"Accuracy: {accuracy_score(y_test, predictions_clean_test):.8f}")
print(f"Precision:{precision_score(y_test, predictions_clean_test, average='weighted'):.8f}")
print(f"Recall: {recall_score(y_test, predictions_clean_test, average='weighted'):.8f}")
print(f"F1 score: {f1_score(y_test, predictions_clean_test, average='weighted'):.8f}")
print(f"\nClassification Report:\n {classification_report(y_test, predictions_clean_test)}")
print(f"\nConfusion Matrix:\n {confusion_matrix(y_test, predictions_clean_test)}")

print(f"Runtime: {(time.time()-st):.1f}")



