<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

<h1 align="center">🛡️ SpamGuard AI</h1>
<p align="center">
  <b>Advanced SMS & Email Spam Detection Pipeline</b><br>
  <i>Classifying malicious text using Natural Language Processing and Machine Learning</i>
</p>

---

## 📝 Overview

**SpamGuard AI** is an end-to-end Machine Learning web application designed to automatically classify incoming messages as **Spam** (malicious, phishing, or advertising) or **Ham** (legitimate). 

Built using Python, `scikit-learn`, and `NLTK`, this project encompasses the complete data science lifecycle—from exploratory data analysis and text preprocessing to model training and deployment via a live **Streamlit** dashboard.

## 🚀 Key Features

* **Advanced NLP Preprocessing:** Intelligent tokenization, HTML stripping, punctuation/stopword removal, and Porter Stemming.
* **Algorithm Benchmarking:** Compares multiple classifiers including:
  - Multinomial Naive Bayes *(Best Performer)*
  - Support Vector Machines (SVM)
  - Random Forest
  - Logistic Regression
* **Feature Engineering:** Uses Term Frequency-Inverse Document Frequency (TF-IDF) statistical tracking (up to 3,000 max features).
* **High Accuracy:** Achieved **~97.4% Accuracy** and **100% Precision** on test splits, ensuring zero false-positive spam blocks.
* **Interactive UI:** A real-time web application deployed locally via Streamlit for instant predictions.

## 📁 Repository Structure

```text
SpamGuard-AI/
├── app/
│   └── app.py                  # Streamlit web application dashboard
├── notebooks/
│   └── Spam_Detection.ipynb    # Full exploratory data analysis & algorithm testing
├── README.md                   # Project documentation
├── requirements.txt            # Dependency configuration
├── train.py                    # Standalone executable python training pipeline
├── model.pkl                   # Trained Multinomial Naive Bayes model (Binary)
└── vectorizer.pkl              # TF-IDF trained vocabulary vectorizer (Binary)
```

## ⚙️ Installation & Usage

To run this project locally, you will need Python 3.8+ installed.

**1. Clone the repository:**
```bash
git clone https://github.com/pendalwarmadhukar/SpamGuard-AI.git
cd SpamGuard-AI
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train the model (Optional):**
If you wish to re-train the model or generate the `.pkl` files yourself, run the training pipeline. It will automatically download the required UCI dataset.
```bash
python train.py
```

**4. Start the Web Dashboard:**
Launch the Streamlit app to test custom SMS or email messages.
```bash
streamlit run app/app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`*

## 📊 Evaluation Metrics
*The primary model (`MultinomialNB`) evaluated on a 20% test split from the UCI SMS Spam Collection Dataset.*

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **Accuracy** | 97.4% | Overall correctness across all classifications. |
| **Precision** | 100.0% | Out of all predicted spam, 100% were actually spam (*Flawless*). |
| **Recall** | ~80.7% | Ability of the model to find all the positive spam cases. |
| **F1-Score** | 0.89 | Harmonic mean of Precision and Recall. |

## 🔮 Future Enhancements
- [ ] **Transformer Models:** Upgrade the underlying engine from NB to DistilBERT or RoBERTa for deep contextual understanding.
- [ ] **Multilingual Support:** Implement spaCy text processing to classify spam in non-English languages.
- [ ] **Enron Integration:** Expand the training dataset with the massive Enron Email corpus to identify corporate-level phishing attacks.
- [ ] **Gmail API:** Connect the script directly to a live email inbox via OAuth2 to auto-label and filter incoming messages.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

---
<p align="center">Developed for Cyber Security & Machine Learning Portfolios</p>
