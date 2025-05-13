# Fake News Detection Using Logistic Regression and NLP
This project demonstrates a machine learning pipeline to classify news articles as fake or real using a combination of natural language processing (NLP) techniques and a Logistic Regression classifier.

# 🔍 Dataset
The dataset includes two CSV files:
Fake.csv: Contains fake news articles.
True.csv: Contains real news articles.
These files are merged and labeled (1 for fake, 0 for real).

# ⚙️ Workflow
Text Cleaning: Lowercasing, contraction handling, punctuation removal, stopword filtering, and lemmatization using spaCy and NLTK.
Feature Extraction: TF-IDF vectorization with unigrams and bigrams (max_features=5000).
Model: Logistic Regression trained on 67% of the dataset.
Evaluation:
Accuracy: 98.79%
F1-score: 0.99
Confusion Matrix: Shows very low false positives/negatives.

# 📈 Results
Accuracy: 0.9879
Classification Report:
               precision    recall  f1-score   support
           0       0.98      0.99      0.99      7039
           1       0.99      0.99      0.99      7778
Confusion Matrix:
 [[6974   65]
  [ 113 7665]]

# 📦 Libraries Used
pandas, numpy, matplotlib, seaborn
nltk, spacy
sklearn: TF-IDF, Logistic Regression, metrics
re for regex-based text cleaning

# 🛠️ Future Work
Upgrade to deep learning models like BERT for better semantic understanding.
Include subject classification as a multi-label task.
Deploy using a simple web interface for real-time news checks.

