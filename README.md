# 💳 Credit Card Fraud Detection System

A machine learning system that detects fraudulent transactions in real-time.  
The system is built using **XGBoost** and **Random Forest**, combined into an **ensemble model**, and served through a **Flask web app**.

---

## 🚀 Project Overview

Credit card fraud is a critical problem in the financial sector.  
Fraudulent transactions are **rare (imbalanced dataset)**, making detection challenging.  

This project demonstrates:
- Handling **imbalanced data**
- Using **multiple models** (XGBoost + Random Forest)
- **Threshold tuning** to improve recall (catching more frauds)
- A **Flask-based web app** to:
  - Enter transaction data (`Amount`, `Time`)
  - Auto-generate anonymized features (`V1–V28`)
  - Display prediction probability (% fraud)
  - Show fraud detection results interactively

---

## 📊 Dataset

We use the **Kaggle Credit Card Fraud Dataset**:

- **Time**: Seconds elapsed since first transaction
- **Amount**: Transaction amount
- **V1–V28**: Anonymized features (from PCA transformation)
- **Class**: Target variable  
  - `0` = Not Fraud  
  - `1` = Fraud

👉 Only ~0.17% of all transactions are fraudulent, making this an **imbalanced classification problem**.

---

## 🧠 Models

### 1. **XGBoost Classifier**
- Great at handling class imbalance
- Higher **recall** (better at catching frauds)

### 2. **Random Forest Classifier**
- More conservative
- Higher **precision** (fewer false alarms)

### 3. **Ensemble Model**
- We combine both models:
  - `avg_proba = 0.6 * proba_xgb + 0.4 * proba_rf`
- Then apply a **tuned threshold** (0.35 instead of default 0.5)
- This balances **recall and precision**, improving fraud detection

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Drop target (`Class`) for feature median storage
   - Handle imbalance
   - Train-test split

2. **Training**
   - Train XGBoost and Random Forest separately
   - Save models with `joblib`

3. **Ensembling**
   - Weighted average of probabilities
   - Adjust threshold for higher recall

4. **Deployment (Flask App)**
   - User inputs `Amount` and `Time`
   - `V1–V28` auto-filled with median/randomized values
   - Ensemble model predicts fraud probability
   - Result displayed in the browser

---

## 📈 Performance

Confusion Matrix (Ensemble, weighted avg, threshold = 0.35):

[[56827 37]
[ 12 86]]


Classification Report:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Not Fraud) | 1.00 | 1.00 | 1.00 | 56864 |
| 1 (Fraud)     | 0.70 | 0.88 | 0.78 | 98 |

- **Accuracy:** 1.00  
- **Macro Avg F1:** 0.89  
- **Weighted Avg F1:** 1.00  

👉 The model catches **88% of all frauds** while keeping very few false positives.

---

## 🖥️ Flask Web App

Features:
- Enter `Amount` and `Time`
- Click **"Predict"** → Fraud probability shown
- Click **"View Random V1–V28"** → See anonymized features in a table

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/Credit-Card-Fraud-Detection-system.git
cd Credit-Card-Fraud-Detection-system
pip install -r requirements.txt
```
▶️ Run the App
```bash
python app.py
```
Then open your browser at http://127.0.0.1:5000/


📜 License

MIT License – free to use and modify.
