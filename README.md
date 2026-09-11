# 💬 Customer Review Sentiment Analysis Using Machine Learning

## 📌 Project Overview

Customer Review Sentiment Analysis is a Natural Language Processing (NLP) and Machine Learning project that automatically identifies whether a customer review expresses a **Positive** or **Negative** sentiment.

The project processes customer review text, converts it into numerical features using TF-IDF, and applies Machine Learning classification algorithms to predict sentiment.

## 🎯 Objective

The main objective of this project is to build a Machine Learning model that can automatically classify customer reviews as:

- 😊 Positive
- 😞 Negative

## 📊 Dataset

The project uses Amazon customer review data containing **1,000 reviews**.

- 500 Positive Reviews
- 500 Negative Reviews

### Target Variable

`Sentiment`

- `1` → Positive Review
- `0` → Negative Review

## 🧹 Text Preprocessing

The review text was prepared before training the Machine Learning models.

The main preprocessing steps included:

- Checking the dataset
- Checking for missing values
- Cleaning review text
- Preparing the text for feature extraction

## 🔤 Feature Extraction

### TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert customer review text into numerical features that Machine Learning algorithms can understand.

The TF-IDF vectorizer was configured with a maximum of **5,000 features**.

## 🤖 Machine Learning Models

Two classification algorithms were tested:

### 1. Logistic Regression

Accuracy:

**79.50%**

Confusion Matrix:

```text
[[82, 18],
 [23, 77]]
```

### 2. Multinomial Naive Bayes

Accuracy:

**80.50%**

🏆 **Multinomial Naive Bayes achieved the best accuracy and was selected as the final model.**

## 🔄 Project Workflow

```text
Customer Reviews
       ↓
Text Cleaning
       ↓
TF-IDF Vectorization
       ↓
Train-Test Split
       ↓
Machine Learning Models
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Sentiment Prediction
```

## 🖥️ Streamlit Application

A Streamlit web application was developed to make the model easy to use.

Users can enter a customer review, and the application predicts whether the sentiment is:

- 😊 Positive
- 😞 Negative

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Streamlit
- Jupyter Notebook

## 🚀 Future Improvements

- Train the model using a larger review dataset
- Add Neutral sentiment classification
- Experiment with advanced NLP techniques
- Try Deep Learning models such as LSTM
- Improve the Streamlit user interface
- Deploy the application online

## 👨‍💻 Author

**K.P. Vishwa**

Aspiring Data Scientist | Data Analyst | Machine Learning Enthusiast
