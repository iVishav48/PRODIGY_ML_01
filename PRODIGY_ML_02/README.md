# Customer Segmentation using K-Means Clustering

## 📌 Project Overview
This project performs **customer segmentation** using the **K-Means clustering algorithm**.  
The goal is to group mall customers based on their **annual income** and **spending behavior** in order to identify distinct customer segments.

This is an **unsupervised machine learning** project, where no target labels are provided and patterns are discovered automatically.

---

## 📂 Dataset
The dataset used in this project is obtained from **Kaggle**:

🔗 **Kaggle Dataset Link**:  
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

**File used:**  
- `Mall_Customers.csv`

---

## 📊 Features Used
The following features were selected for clustering:
- **Annual Income (k$)**
- **Spending Score (1–100)**

These features represent customer purchasing power and spending behavior.

---

## ⚙️ Algorithm Used
- **K-Means Clustering**
- **Elbow Method** was used to determine the optimal number of clusters.

---

## 🛠️ Technologies & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## ▶️ How to Run the Project
1. Download the dataset from Kaggle.
2. Place `Mall_Customers.csv` in the project directory.
3. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
