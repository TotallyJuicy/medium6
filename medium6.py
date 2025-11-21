# ============================================
# Steam Review Sentiment Classifier (Module 6)
# WITH FILE ACCESS TEST
# ============================================

import os
import pandas as pd
import numpy as np

# --- FILE ACCESS DEBUG TEST ---
print("=== DEBUG: WORKING DIRECTORY ===")
print(os.getcwd())

print("\n=== DEBUG: LIST OF FILES IN THIS FOLDER ===")
for f in os.listdir():
    print(" -", f)

print("\n=== DEBUG: ATTEMPTING TO OPEN dataset.csv ===")

try:
    df_test = pd.read_csv(
        "dataset.csv",
        nrows=5,
        engine="python",
        encoding_errors="ignore"
    )
    print("SUCCESS OPENING FILE!")
    print(df_test.head())
except Exception as e:
    print("FAILED TO OPEN FILE:")
    print(e)
    print("\nIf this fails, Python is not seeing dataset.csv in this folder.")
    print("Fix = run your script from the correct folder or use full file path.")
    exit()   # stop script if file can't be opened
# --- END FILE ACCESS TEST ---


# ============================================
# 1. LOAD SAFE SAMPLE FROM HUGE DATASET
# ============================================

df = pd.read_csv(
    "dataset.csv",
    nrows=50000,                    # safe limit
    engine="python",
    encoding_errors="ignore"
)

# ============================================
# 2. CLEANING
# ============================================

df = df[['review_text', 'review_score']]  # keep only needed columns
df = df.dropna(subset=['review_text'])    # drop empty reviews

# Convert sentiment -1/1 to 0/1
df['label'] = df['review_score'].apply(lambda x: 1 if x == 1 else 0)

# Remove tiny reviews
df = df[df['review_text'].str.len() > 5]

print("Dataset size after cleaning:", len(df))

# ============================================
# 3. TRAIN/TEST SPLIT
# ============================================

from sklearn.model_selection import train_test_split

X = df['review_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 4. TF–IDF
# ============================================

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=20000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ============================================
# 5. MODELS
# ============================================

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

nb_model = MultinomialNB().fit(X_train_tfidf, y_train)
knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train_tfidf, y_train)
dt_model = DecisionTreeClassifier(max_depth=20, random_state=42).fit(X_train_tfidf, y_train)

# ============================================
# 6. PREDICTIONS
# ============================================

nb_pred = nb_model.predict(X_test_tfidf)
knn_pred = knn_model.predict(X_test_tfidf)
dt_pred = dt_model.predict(X_test_tfidf)

# ============================================
# 7. EVALUATION
# ============================================

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

def evaluate_model(name, y_true, y_pred):
    print("\n" + "=" * 60)
    print(f"{name} RESULTS")
    print("=" * 60)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

evaluate_model("Naive Bayes", y_test, nb_pred)
evaluate_model("KNN", y_test, knn_pred)
evaluate_model("Decision Tree", y_test, dt_pred)

# ============================================
# 8. ROC CURVE (NB)
# ============================================

nb_proba = nb_model.predict_proba(X_test_tfidf)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, nb_proba)
roc_auc = auc(fpr, tpr)

print("\n=== ROC AUC (Naive Bayes) ===")
print("AUC Score:", roc_auc)

# ============================================
# 9. MISCLASSIFIED EXAMPLES
# ============================================

test_df = pd.DataFrame({
    'text': X_test,
    'true': y_test,
    'pred': nb_pred
})

wrong = test_df[test_df['true'] != test_df['pred']]

print("\n=== 5 Misclassified Samples ===")
for i in range(5):
    print("\n--- Sample", i+1, "---")
    print("True:", wrong.iloc[i]['true'])
    print("Pred:", wrong.iloc[i]['pred'])
    print("Text:", wrong.iloc[i]['text'])

# ============================================
# DONE
# ============================================
