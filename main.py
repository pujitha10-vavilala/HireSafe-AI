import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import re
import string
import nltk
import warnings
import joblib
import shap

warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

"""nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")"""

print("=" * 50)
print("HIRESAFE AI - FAKE JOB POSTING DETECTION")
print("=" * 50)

#Data Information
df = pd.read_csv("fake_job_postings.csv")
(df.head())
print("Daset Loaded Successfully")

("="*60)
("Shape")
print(f"\nDataset Shape : {df.shape}")
print("\nTarget Classes")
print("-"*25)
print("Legitimate Jobs :", (df["fraudulent"] == 0).sum())
print("Fraudulent Jobs :", (df["fraudulent"] == 1).sum())

"""("="*60)
("Columns")
(df.columns.tolist())

("="*60)
("Info")
(df.info())

("="*60)
("Missing Values")
(df.isnull().sum())

("="*60)
("Duplicates")
(df.duplicated().sum())

("="*60)
("Fraud Distribution")
(df["fraudulent"].value_counts())"""

#Missing Values
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
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
    "salary_range"
]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"\+?\d[\d\s-]{8,}\d", " ", text)

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords and lemmatize
    words = []

    for word in text.split():
        if word not in stop_words:
            word = lemmatizer.lemmatize(word)
            words.append(word)

    return " ".join(words)

for col in text_columns:
    (f"Cleaning {col}...")
    df[col] = df[col].apply(clean_text)

(df["description"].iloc[0])

#Feature Engineering
df["description_length"] = df["description"].apply(len)
df["company_profile_length"] = df["company_profile"].apply(len)
df["requirements_length"] = df["requirements"].apply(len)
df["benefits_length"] = df["benefits"].apply(len)

suspicious_keywords = [
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
    "click"
]

def keyword_score(text):
    text = text.lower()
    score = 0
    for word in suspicious_keywords:
        if word in text:
            score += 1

    return score
df["keyword_score"] = df["description"].apply(keyword_score)

df["missing_company_profile"] = (
    df["company_profile"] == ""
).astype(int)

df["missing_requirements"] = (
    df["requirements"] == ""
).astype(int)

df["missing_benefits"] = (
    df["benefits"] == ""
).astype(int)

(df[[
    "description_length",
    "company_profile_length",
    "requirements_length",
    "benefits_length",
    "keyword_score",
    "missing_company_profile",
    "missing_requirements"
]].head())

df["combined_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)
(df["combined_text"].iloc[0][:1000])

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)
X_text = tfidf.fit_transform(df["combined_text"])
(X_text.shape)

#Vectorize Numerical Columns
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
    "has_questions"
]

X_numeric = df[numeric_features]

X = hstack([X_text, X_numeric.values])
y = df["fraudulent"]
(X.shape)
(y.shape)

#Train-test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
stratify=y

#Model Training
models = {
    "Logistic Regression": LogisticRegression(
    max_iter=1000,
    class_weight="balanced"),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(
    class_weight="balanced",
    random_state=42
),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = []
for name, model in models.items():
    print("=" * 60)
    print(name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
    print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results)
print(results_df.sort_values("F1", ascending=False))

#Visualizations

#Target variable
counts = df["fraudulent"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(["Legitimate", "Fraudulent"], counts.values)
plt.title("Distribution of Job Postings")
plt.ylabel("Count")

for i, v in enumerate(counts.values):
    plt.text(i, v+100, str(v), ha="center")

plt.show()

#Description length
plt.figure(figsize=(8,5))
plt.hist(df["description_length"], bins=40)
plt.xlim(0,3000)
plt.xlabel("Description Length")
plt.ylabel("Frequency")
plt.title("Distribution of Job Description Length")

plt.show()

#Model Comparision
metrics = ["Accuracy", "Precision", "Recall", "F1"]

x = np.arange(len(results_df))
width = 0.2

plt.figure(figsize=(12,6))

for i, metric in enumerate(metrics):
    plt.bar(
        x + i*width,
        results_df[metric],
        width,
        label=metric
    )

plt.xticks(
    x + width*1.5,
    results_df["Model"],
    rotation=15
)

plt.ylim(0,1.05)
plt.ylabel("Score")
plt.title("Performance Comparison of Machine Learning Models")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

#Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
best_model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

best_model.fit(X_train, y_train)

disp = ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    display_labels=["Legitimate", "Fraudulent"],
    cmap="Blues",
    values_format="d"
)

disp.ax_.set_title("Confusion Matrix - Linear SVM")

plt.tight_layout()
plt.show()

#ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get decision scores
y_score = best_model.decision_function(X_test)

# Compute ROC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    color="darkblue",
    linewidth=3,
    label=f"Linear SVM (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    '--',
    color="red",
    linewidth=2,
    label="Random Guess"
)

plt.xlim([0,1])
plt.ylim([0,1.05])

plt.xlabel("False Positive Rate",fontsize=12)
plt.ylabel("True Positive Rate",fontsize=12)

plt.title("ROC Curve of Linear SVM",fontsize=15,fontweight="bold")
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()

plt.show()


# Numeric features used in your model
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
    "has_questions"
]

# Coefficients corresponding only to numeric features
numeric_coef = best_model.coef_[0][-len(numeric_features):]

importance = pd.DataFrame({
    "Feature": numeric_features,
    "Importance": np.abs(numeric_coef)
})

# Top 5 features
top5 = importance.sort_values(
    by="Importance",
    ascending=False
).head(5)

# Plot
plt.figure(figsize=(8,5))

plt.barh(top5["Feature"], top5["Importance"])

plt.xlabel("Coefficient Magnitude")
plt.ylabel("Feature")
plt.title("Top 5 Most Important Engineered Features")
plt.gca().invert_yaxis()

# Display values on bars
for i, value in enumerate(top5["Importance"]):
    plt.text(value, i, f"{value:.3f}", va="center")

plt.tight_layout()
plt.show()

#Hypermeter Tuning
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}
grid = GridSearchCV(
    estimator=LinearSVC(
        class_weight="balanced",
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("="*50)
print("Hyperparameter Tuning Results")
print("="*50)

print("Best Parameters :", grid.best_params_)
print("Best CV F1 Score :", round(grid.best_score_,4))
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
before_f1 = 0.8448   # Your current F1

after_f1 = f1_score(y_test, y_pred)

print("\nBefore Tuning :", round(before_f1,4))
print("After Tuning  :", round(after_f1,4))
joblib.dump(best_model, "best_linear_svm.pkl")
print("Tuned model saved successfully!")

# Save trained model
joblib.dump(best_model, "linear_svm_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Model and Vectorizer saved successfully!")

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
    "has_questions"
]

joblib.dump(numeric_features, "numeric_features.pkl")

#Explainable AI
sample = 5
prediction = best_model.predict(X_test[sample])[0]
print("\n Prediction")
print("-------------------------")

if prediction == 0:
    print(" Legitimate Job Posting")
else:
    print(" Fraudulent Job Posting")

score = best_model.decision_function(X_test[sample])[0]

confidence = 100 / (1 + np.exp(-abs(score)))

print("\n AI Confidence Score")
print("-------------------------")
print(f"{confidence:.2f}%")

print("\n Safety Tips")
print("-------------------------")

if prediction == 1:
    print("✔ Verify the company's official website.")
    print("✔ Never pay registration or recruitment fees.")
    print("✔ Check company reviews before applying.")
    print("✔ Contact the recruiter through official channels.")
else:
    print("✔ This job appears legitimate.")
    print("✔ Still verify the employer before sharing personal information.")

print("\n Similar Job Search")
print("-------------------------")

job_title = df.iloc[sample]["title"]
print("LinkedIn : https://www.linkedin.com/jobs/search/?keywords=" + job_title.replace(" ","+"))
print("Indeed   : https://in.indeed.com/jobs?q=" + job_title.replace(" ","+"))
print("Naukri   : https://www.naukri.com/" + job_title.replace(" ","-").lower() + "-jobs")

print("\n Suspicious Keywords")
print("-------------------------")

fake_keywords = [
    "urgent",
    "easy money",
    "registration fee",
    "work from home",
    "investment",
    "click here",
    "guaranteed",
    "quick cash",
    "limited seats",
    "earn daily"
]

text = (
    str(df.iloc[sample]["title"]) + " " +
    str(df.iloc[sample]["description"])
).lower()

found = False

for word in fake_keywords:
    if word in text:
        print("⚠", word)
        found = True

if not found:
    print("No suspicious keywords detected.")
