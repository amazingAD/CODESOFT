# 🎬 Movie Genre Classification using NLP

This project builds a text classification system to predict the **genre** of a movie based on its **description** using Natural Language Processing (NLP) and machine learning models.

---

## 🗂️ Dataset Overview

The dataset consists of two primary files:

- `train_data.txt`: Contains labeled movie data with fields:

- `test_data.txt`: Contains unlabeled data for prediction:

- `test_data_solution.txt`: (Optional) Contains ground truth for evaluation.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Regex & Text Cleaning
- TF-IDF Vectorization

---

## 🔍 Models Trained

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Support Vector Machine (LinearSVC)**

The best-performing model (Logistic Regression) is used for the final predictions.

---

## 📖 Preprocessing Steps

1. **Text Cleaning**  
 - Lowercasing
 - Removing special characters using regex

2. **Vectorization**  
 - TF-IDF (`max_features=5000`)
 - Removed English stopwords

3. **Train-Test Split**  
 - 80% training, 20% validation from training data

---

## 🧠 Model Performance (Validation)

| Model               | Accuracy | Notes |
|--------------------|----------|-------|
| Naive Bayes         | ~        | Fast but lower performance  
| Logistic Regression | ~        | Best results overall  
| SVM (Linear)        | ~        | Competitive, slower than LR  

> 🔍 Fill in actual accuracy after running.

---

## 📁 Output

- A CSV file `genre_predictions.csv` is generated containing:


---

## 🧪 Final Evaluation (Optional)

If `test_data_solution.txt` is provided:
- The predictions are compared with true genres.
- Accuracy and classification report are displayed.

---

## 🚀 How to Run

1. Install required packages:

```bash
pip install pandas numpy scikit-learn

movie_genre_classifier/
│
├── train_data.txt
├── test_data.txt
├── test_data_solution.txt       # Optional
├── movie_genre_classifier.py    # Or .ipynb
├── genre_predictions.csv
└── README.md

👨‍💻 Author
ADITYA KUMAR
