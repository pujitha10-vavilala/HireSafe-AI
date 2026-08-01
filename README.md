#  JobGuard AI – Fake Job Posting Detection using Machine Learning

##  Project Overview

JobGuard AI is a Machine Learning project that detects fraudulent job postings using Natural Language Processing (NLP) and supervised learning algorithms.

The system analyzes job descriptions, company profiles, requirements, and other job-related information to classify a job posting as **Legitimate** or **Fraudulent**.

---

##  Features

- Data Cleaning & Preprocessing
- NLP Text Cleaning
- TF-IDF Vectorization
- Feature Engineering
- Model Comparison
- Hyperparameter Tuning
- AI Confidence Score
- Explainable AI
- Model Saving
- Fraud Detection

---

##  Dataset

**Source:** Kaggle

Dataset Name:
Fake Job Postings Dataset

Number of Records:
- Total Jobs : 17,880

Target Classes:
- Legitimate Jobs : 17,014
- Fraudulent Jobs : 866

---

##  Data Preprocessing

- Removed missing values
- Text cleaning
- Lowercase conversion
- Stopword removal
- Lemmatization
- Feature Engineering
- TF-IDF Vectorization

---

##  Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Naive Bayes
- Linear SVM
- Random Forest

---

##  Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Linear SVM | **98.49%** | **84.00%** | **84.97%** | **84.48%** |
| Random Forest | 98.21% | 100% | 63.00% | 77.30% |
| Logistic Regression | 92.05% | 36.94% | 90.75% | 52.51% |
| Naive Bayes | 65.10% | 9.25% | 70.52% | 16.35% |

---

##  Best Model

**Linear Support Vector Machine (Linear SVM)**

Reasons for selection:

- Highest F1 Score
- Excellent Precision
- Excellent Recall
- Handles high-dimensional TF-IDF features efficiently

---

##  Hyperparameter Tuning

GridSearchCV was used with 5-Fold Cross Validation.

Best Parameter

```
C = 10
```

Improved F1 Score

```
Before Tuning : 0.8448

After Tuning : 0.8536
```

---

##  Project Visualizations

- Target Class Distribution
- Job Description Length Distribution
- Model Comparison
- Confusion Matrix
- ROC Curve
- Feature Importance

---

##  Project Structure

```
JobGuard-AI
│
├── Images
├── results
├── models
├── fake_job_postings.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- NLTK
- Joblib

---

##  Future Enhancements

- Streamlit Web Application
- Real-time Job Fraud Detection
- Explainable AI Dashboard
- AI-based Safety Recommendations
- Resume-Job Matching

---

##  Author

**Pujitha Vavilala**

Computer Science Engineering Student

Machine Learning & Artificial Intelligence Enthusiast