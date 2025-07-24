# 🛡️ Credit Card Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent transactions using supervised machine learning techniques on a real-world dataset. Three popular models are trained and compared: Logistic Regression, Decision Tree, and Random Forest.

---

## 📂 Dataset

The data is split into two CSV files:
- `fraudTrain.csv` — training data  
- `fraudTest.csv` — testing data

Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

---

## 🧠 Models Used

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

---

## ⚙️ Preprocessing Steps

1. **Dropped Irrelevant Columns**  
   Columns such as personal identifiers, transaction time, address, and high-cardinality strings were removed to avoid overfitting and leakage.

2. **One-Hot Encoding**  
   - Categorical columns: `category`, `gender`
   - Used `pd.get_dummies(..., drop_first=True)` to avoid multicollinearity.

3. **Feature Scaling**  
   - Applied `StandardScaler` to normalize features for all models.

---

## 🧪 Evaluation Metrics

- **Accuracy Score**
- **ROC AUC Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix**

---

## 📊 Results

| Model              | Accuracy | ROC AUC |
|--------------------|----------|---------|
| Logistic Regression | ~       | ~       |
| Decision Tree       | ~       | ~       |
| Random Forest       | ~       | ~       |

> 🔍 _You can fill in the exact values based on your output._

---

## 🖼️ Visualization

- A confusion matrix is displayed for the best-performing model (Random Forest).

---

## 🏁 How to Run

1. Install required libraries (if not already installed):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

fraud_detection_project/
│
├── fraudTrain.csv
├── fraudTest.csv
├── fraud_detection.ipynb
└── README.md

author--

ADITYA KUMAR