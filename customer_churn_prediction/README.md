# Customer Churn Prediction 🔍📉

This project focuses on building a Machine Learning model to predict customer churn for a subscription-based service using historical customer data.

## 📁 Project Description

Customer churn refers to when a customer stops doing business with a company. Predicting churn allows businesses to take proactive actions and improve customer retention.

In this project, we used classification models such as:
- Logistic Regression
- Random Forest
- Gradient Boosting

to predict whether a customer is likely to churn (`Exited = 1`) or not (`Exited = 0`).

---

## 📊 Dataset

- File: `Churn_Modelling.csv`
- Source: Banking customer data
- Features include:
  - CreditScore, Geography, Gender, Age, Tenure
  - Balance, Number of Products, HasCrCard, IsActiveMember, EstimatedSalary
  - Exited (Target variable)

---

## 🧠 Models Used

| Model                  | Description                              |
|-----------------------|------------------------------------------|
| Logistic Regression    | Simple baseline classifier               |
| Random Forest Classifier | Ensemble-based model using bagging     |
| Gradient Boosting Classifier | Ensemble-based model using boosting |

---

## 📈 Results Summary

| Model        | Accuracy (on Test Set) |
|--------------|------------------------|
| Logistic Regression  | ~83%              |
| Random Forest        | ~86%              |
| Gradient Boosting    | ~87% (Best)       |

✅ **Gradient Boosting performed the best** in terms of accuracy and AUC-ROC score.

---

## 📉 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC-AUC Curve
- Feature Importance (for Gradient Boosting)

---

## 📦 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
