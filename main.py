import warnings
import re
import string
import joblib
import nltk
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# NLTK
# ============================================================

for package in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(package, quiet=True)
    except Exception:
        pass

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ============================================================
# TEXT PREPROCESSING
# IMPORTANT: app.py uses this same logic.
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s-]{8,}\d", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)


# ============================================================
# DATASET
# ============================================================

df = pd.read_csv("fake_job_postings.csv")

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

for col in text_columns:
    df[col] = df[col].fillna("")

categorical_columns = [
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location",
    "salary_range",
]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

print("=" * 60)
print("HIRESAFE AI - FAKE JOB POSTING DETECTION")
print("=" * 60)
print("Dataset shape:", df.shape)
print("Legitimate jobs:", (df["fraudulent"] == 0).sum())
print("Fraudulent jobs:", (df["fraudulent"] == 1).sum())

# Apply the exact same preprocessing that the deployed app will use.
for col in text_columns:
    df[col] = df[col].apply(clean_text)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

SUSPICIOUS_KEYWORDS = [
    "urgent",
    "immediate",
    "quick",
    "easy",
    "earn",
    "income",
    "bonus",
    "limited",
    "guaranteed",
    "apply now",
    "work from home",
    "no experience",
    "investment",
    "payment",
    "registration fee",
    "click",
]


def keyword_score(text):
    text = str(text).lower()
    return sum(1 for word in SUSPICIOUS_KEYWORDS if word in text)


df["description_length"] = df["description"].apply(len)
df["company_profile_length"] = df["company_profile"].apply(len)
df["requirements_length"] = df["requirements"].apply(len)
df["benefits_length"] = df["benefits"].apply(len)
df["keyword_score"] = df["description"].apply(keyword_score)
df["missing_company_profile"] = (df["company_profile"] == "").astype(int)
df["missing_requirements"] = (df["requirements"] == "").astype(int)
df["missing_benefits"] = (df["benefits"] == "").astype(int)

# EXACT feature order used by app.py.
numeric_features = [
    "description_length",
    "company_profile_length",
    "requirements_length",
    "benefits_length",
    "keyword_score",
    "missing_company_profile",
    "missing_requirements",
    "missing_benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

# ============================================================
# TEXT FEATURES
# ============================================================

df["combined_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
).str.strip()

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
)

X_text = tfidf.fit_transform(df["combined_text"])
X_numeric = df[numeric_features].values

X = hstack([X_text, X_numeric])
y = df["fraudulent"]

print("TF-IDF features:", X_text.shape[1])
print("Total model features:", X.shape[1])

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# ============================================================
# MODEL COMPARISON
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
}

results = []

for name, model in models.items():
    print("\n" + "=" * 60)
    print(name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    })

    print(classification_report(y_test, y_pred, zero_division=0))

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# ============================================================
# HYPERPARAMETER TUNING - LINEAR SVM
# ============================================================

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    estimator=LinearSVC(
        class_weight="balanced",
        random_state=42,
    ),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n" + "=" * 60)
print("FINAL LINEAR SVM")
print("Best parameters:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 4))
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Test precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Test recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("Test F1:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# SAVE DEPLOYMENT ARTIFACTS
# ============================================================

joblib.dump(best_model, "best_linear_svm.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(numeric_features, "numeric_features.pkl")
joblib.dump(SUSPICIOUS_KEYWORDS, "suspicious_keywords.pkl")

print("\nSaved:")
print("- best_linear_svm.pkl")
print("- tfidf_vectorizer.pkl")
print("- numeric_features.pkl")
print("- suspicious_keywords.pkl")

# ============================================================
# SIMPLE EXPLAINABILITY CHECK
# ============================================================

sample = 5
sample_score = float(best_model.decision_function(X_test[sample])[0])
sample_prediction = int(best_model.predict(X_test[sample])[0])

sample_confidence = 100 / (1 + np.exp(-abs(sample_score)))

print("\nEXPLAINABILITY CHECK")
print("Prediction:", "Fraudulent" if sample_prediction == 1 else "Legitimate")
print("Confidence estimate:", round(sample_confidence, 2), "%")
print("Numeric features used:")
print(numeric_features)
