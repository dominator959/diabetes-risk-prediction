# 🩺 Diabetes Risk Prediction

A classification project predicting whether a patient is at risk of diabetes based on clinical measurements. Built using the **Pima Indians Diabetes dataset** from the UCI Machine Learning Repository.

---

## 📊 Problem Statement

Given eight medical measurements for a female patient, predict whether she is diabetic or not. This is a binary classification problem with real-world healthcare applications.

---

## 📁 Dataset

- **Source**: UCI Machine Learning Repository via Kaggle  
- **Records**: 768 patients  
- **Features**: 8 clinical measurements  
- **Target**: Outcome (1 = diabetic, 0 = not diabetic)  
- **Class balance**: 65% non-diabetic, 35% diabetic  

---

## 🗂️ Project Structure
```text
diabetes-risk-prediction/
├── 📂 data/
│   ├── 📁 raw/               # original unmodified dataset
│   └── 📁 processed/         # cleaned dataset produced by notebooks
├── 📓 notebooks/
│   ├── 📄 01_data_loading.ipynb  # load data, validate schema, inspect 0s
│   ├── 📄 02_data_cleaning.ipynb # replace invalid 0s, impute missing val
│   ├── 📄 03_eda.ipynb           # distributions, correlations, boxplots
│   └── 📄 04_modeling.ipynb      # train models, evaluate, feature importance
├── 💻 src/
│   ├── 🐍 data_loader.py     # reusable loading and validation functions
│   ├── 🐍 preprocessor.py    # cleaning and feature engineering functions
│   └── 🐍 evaluator.py       # metrics and plotting functions
├── 📈 outputs/
│   ├── 🖼️ figures/           # all saved charts
│   └── 📑 reports/
├── 📝 requirements.txt
└── 📖 README.md
```
---

## 🔍 Key Findings

- 🔑 **Insulin** is the most important feature according to Random Forest  
- 📈 **Glucose** has the highest correlation with the target (0.50)  
- 👩‍⚕️ Diabetic patients show noticeably higher **Glucose**, **BMI**, and **Age**  
- ⚠️ **SkinThickness** and **Insulin** had the most missing data (29.6% and 48.7%)

---

## 📈 Model Results

| Model               | ROC AUC |
|---------------------|---------|
| 🌲 Random Forest    | 0.9447  |
| 🌳 Decision Tree    | 0.9121  |
| 📉 Logistic Regression | 0.8263  |

> Random Forest performed best with an AUC of **0.9447** and also achieved the highest accuracy.

---

## 🧹 Data Cleaning Approach

Several columns contained zeros that are medically impossible (a person cannot have a BMI of 0 or a glucose level of 0). These were replaced with `NaN` and filled using the **median of each group separately** — diabetic patients were imputed from the diabetic group median and non‑diabetic patients from their own group median. This preserves the statistical difference between the two groups rather than collapsing them into a single population median.

---

## 🚀 How to Run

1. Clone this repository  
2. Install requirements:  
   ```bash
   pip install -r requirements.txt
3. Place diabetes.csv in data/raw/
4. Run notebooks in order:
   ```bash
   O1_data_loading → O2_data_cleaning → O3_eda → O4_modeling

---

## 🛠️ Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit‑learn, Jupyter Notebook

---

## 👨‍💻 Author
Muhammad Usman — BS Data Science Student
🔗 LinkedIn https://www.linkedin.com/in/muhammad-usman-157841269/
🐙 GitHub https://github.com/dominator959

