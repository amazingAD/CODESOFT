# %%
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")


# %%
# Load and parse training data
train_data = []
with open("train_data.txt", "r", encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) == 4:
            ID, title, genre, description = parts
            train_data.append((ID, title, genre, description))

train_df = pd.DataFrame(train_data, columns=["ID", "Title", "Genre", "Description"])
print("Train shape:", train_df.shape)
train_df.head()


# %%
# Load and parse test data
test_data = []
with open("test_data.txt", "r", encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) == 3:
            ID, title, description = parts
            test_data.append((ID, title, description))

test_df = pd.DataFrame(test_data, columns=["ID", "Title", "Description"])
print("Test shape:", test_df.shape)
test_df.head()


# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

train_df['Clean_Description'] = train_df['Description'].apply(clean_text)
test_df['Clean_Description'] = test_df['Description'].apply(clean_text)


# %%
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(train_df['Clean_Description'])
X_test_final = vectorizer.transform(test_df['Clean_Description'])

y = train_df['Genre']


# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_val)

print("Naive Bayes Accuracy:", accuracy_score(y_val, nb_preds))
print(classification_report(y_val, nb_preds))


# %%
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_val)

print("Logistic Regression Accuracy:", accuracy_score(y_val, lr_preds))
print(classification_report(y_val, lr_preds))


# %%
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_val)

print("SVM Accuracy:", accuracy_score(y_val, svm_preds))
print(classification_report(y_val, svm_preds))


# %%
# You can use your best model here — let's use LogisticRegression
final_preds = lr_model.predict(X_test_final)

# Save results
test_df['Predicted_Genre'] = final_preds
test_df[['ID', 'Title', 'Predicted_Genre']].to_csv("genre_predictions.csv", index=False)
print(" Predictions saved to genre_predictions.csv")


# %%
# Load test solution (optional if you want to evaluate)
solution_data = []
with open("test_data_solution.txt", "r", encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) == 4:
            ID, title, genre, description = parts
            solution_data.append((ID, title, genre, description))

solution_df = pd.DataFrame(solution_data, columns=["ID", "Title", "Genre", "Description"])

# Merge with prediction for evaluation
merged = pd.merge(test_df, solution_df[['ID', 'Genre']], on="ID", how="left", suffixes=('_pred', '_true'))

print("🔍 Final Test Accuracy:", accuracy_score(merged['Genre'], merged['Predicted_Genre']))
print(classification_report(merged['Genre'], merged['Predicted_Genre']))



