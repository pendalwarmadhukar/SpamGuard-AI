# 🛡️ SpamGuard AI: SMS/Email Spam Detection System

An end-to-end Machine Learning project to classify SMS and email messages as spam or ham (legitimate).
Includes a Jupyter Notebook data pipeline for training and an interactive Streamlit web dashboard for real-time predictions.

## 🌟 Features
- **Data Pipeline:** End-to-end preprocessing pipeline built with `scikit-learn` and `NLTK`.
- **Text Preprocessing:** Tokenization, Stopword removal, HTML cleanup, and Porter Stemming.
- **Multiple ML Models:** Trains and compares Naive Bayes, Support Vector Machines, Logistic Regression, and Random Forests.
- **Performance:** Targets 97%+ Accuracy and 95%+ Precision.
- **Web App:** Beautiful Streamlit dashboard to test live emails/texts in real-time.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **NLP Library:** NLTK
- **Machine Learning:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

---

## 🚀 Getting Started

### 1. Installation

First, clone the repository (or navigate to the folder) and install the python dependencies:

```bash
cd "Spam (SMSEmail)"
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline

Before starting the web app, you must train the model. We provide a complete Jupyter Notebook to run the pipeline, which automatically downloads the UCI SMS Spam dataset, cleans it, trains multiple models, evaluates their ROC-AUC curves, and exports the `.pkl` files.

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/Spam_Detection_Pipeline.ipynb`
3. Run all the cells in the notebook.
4. At the end, it will save `model.pkl` and `vectorizer.pkl` into the root directory.

### 3. Start the Web App

Once you have generated the `.pkl` files, you can launch the live Streamlit dashboard.

```bash
streamlit run app/app.py
```
This will open a new tab in your default web browser where you can paste texts and analyze their Spam score.

---

## 📈 Evaluation Metrics

The pipeline automatically evaluates models based on:
- **Accuracy:** Overall correctness.
- **Precision:** Crucial measurement for spam filters (we want a very low False Positive rate so we don't accidentally send important emails to the spam folder).
- **Recall:** Out of all actual spam, how much did we catch.
- **F1-Score:** The harmonic mean of precision and recall.

---

## 🧠 Future Scope / Advanced Extensions

- **Deep Learning (BERT):** The current model utilizes traditional ML with TF-IDF. Exploring transformer architectures like DistilBERT could push accuracy metrics beyond 99% for complex phishing emails.
- **Multilingual Support:** Implement spaCy models to detect spam in Spanish, French, and other languages.
- **Active Learning Feature:** Add a thumbs-up/thumbs-down button in the Streamlit app that logs incorrect predictions locally, allowing the system to be continuously retrained with user feedback.
- **Gmail API Integration:** Deploy the model periodically over unread emails via OAuth2 credentials to auto-label the inbox.
