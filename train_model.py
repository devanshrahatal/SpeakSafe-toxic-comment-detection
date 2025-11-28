import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

from preprocess import clean_text

# ===============================
# 1. Load Dataset
# ===============================

print("Loading dataset...")
df = pd.read_csv("data/train.csv")

# Only keep required columns
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

df = df[["comment_text"] + labels]

# ===============================
# 2. Clean Comments
# ===============================

print("Cleaning text (this may take 1–3 minutes)...")
df["clean_text"] = df["comment_text"].astype(str).apply(clean_text)

# ===============================
# 3. TF-IDF Vectorizer
# ===============================

print("Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=100000,
    stop_words="english",
    ngram_range=(1,2)
)

X = tfidf.fit_transform(df["clean_text"])
y = df[labels].values

# ===============================
# 4. Train/Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. Train Model (SVM One-vs-Rest)
# ===============================

print("Training SVM model...")
model = OneVsRestClassifier(LinearSVC())
model.fit(X_train, y_train)

print("Training complete!")

# ===============================
# 6. Evaluate
# ===============================

print("\nEvaluating model...\n")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=labels))

# ===============================
# 7. Save Model + Vectorizer
# ===============================

print("\nSaving model to /model folder...")

os.makedirs("model", exist_ok=True)

pickle.dump(tfidf, open("model/tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("model/svm_model.pkl", "wb"))

print("Model saved successfully!")
print("Files created:")
print("- model/tfidf_vectorizer.pkl")
print("- model/svm_model.pkl")
