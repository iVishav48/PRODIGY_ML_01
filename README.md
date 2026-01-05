    # House Price Prediction (ML Task 1)

## 📌 Project Overview
This project builds a **machine learning regression model** to predict house prices based on basic house features such as:
- Square footage
- Number of bedrooms
- Number of bathrooms

The goal is to understand the **end-to-end machine learning workflow** by building a simple and interpretable model.

---
## 📊 Dataset Source

The dataset used in this project is from Kaggle:

**House Prices – Advanced Regression Techniques**  
🔗 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The dataset includes:
- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

---

## 🎯 Target Variable
- **SalePrice** → Final selling price of the house

---

## 🧠 Features Used
For simplicity and clarity, only the following features are used:
- `GrLivArea` → Above-ground living area (square feet)
- `BedroomAbvGr` → Number of bedrooms
- `FullBath` → Number of full bathrooms

---

## ⚙️ Machine Learning Approach
- Type: **Supervised Learning**
- Task: **Regression**
- Model Used: **Linear Regression**

---

## 🛠️ Workflow
1. Load training and test data
2. Select required features
3. Handle missing values using median imputation
4. Split training data into train and validation sets
5. Train a Linear Regression model
6. Evaluate model using Mean Absolute Error (MAE)
7. Retrain model on full training data
8. Predict house prices for test data
9. Generate submission file

---

## 📊 Model Evaluation
- Metric Used: **Mean Absolute Error (MAE)**
- MAE represents the average difference between predicted and actual house prices.
- This simple baseline model provides reasonable predictions using limited features.

---

## 📁 Output
- `submission.csv` → Final file containing predicted house prices in the required format

---

## 🧩 Tools & Libraries
- Python
- pandas
- scikit-learn
- VS Code

---

## ✅ Key Learnings
- Understanding train vs test data
- Feature selection
- Data cleaning
- Model training and evaluation
- End-to-end ML pipeline development

---

## 🚀 Conclusion
This project demonstrates a **complete beginner-friendly machine learning workflow**, focusing on clarity, correctness, and practical understanding rather than complex modeling.

