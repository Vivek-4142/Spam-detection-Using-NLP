# 📩 SMS Spam Detection using NLP & Machine Learning

This project is a **Spam Detection System** built using **Natural Language Processing (NLP)** and **Machine Learning**.  
It classifies SMS messages as **Spam** or **Ham (Not Spam)** based on their content.

---

## 🚀 Project Overview

The system analyzes the content of text messages and predicts whether they are spam or legitimate.  
It uses techniques like **text cleaning**, **tokenization**, and **TF-IDF vectorization** combined with machine learning models such as **Logistic Regression** and **Naive Bayes**.

---

## 🧠 Features

- Preprocessing of raw text data (removes noise, stopwords, punctuation, etc.)  
- TF-IDF based feature extraction  
- Trained Machine Learning model for spam detection  
- Interactive **Gradio UI** inside Google Colab (no external URL required)  
- Save and reuse trained model using `joblib`  
- Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix  

---

## 🗂️ Dataset

**Dataset Name:** [SMS Spam Collection Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)  
- 5574 SMS messages  
- Labels:  
  - `ham` → legitimate message  
  - `spam` → unsolicited message  

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`  
  - `gradio` → for building interactive UI in Colab  
  - `joblib` → for saving models  

---

## 🧾 Steps to Run in Google Colab

1. **Clone or upload the project files**  
   ```bash
   !git clone https://github.com/your-username/spam-detection-nlp.git
   cd spam-detection-nlp
