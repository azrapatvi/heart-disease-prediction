# 🫀 Heart Disease Prediction App

A machine learning web app that predicts whether a person has heart disease or not — based on clinical data. Built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 📁 Recommended Repository Name

```
heart-disease-prediction
```

---

## 🗂️ Project Structure

```
heart-disease-prediction/
│
├── heart_disease_prediction.ipynb   ← Jupyter notebook (data analysis + model training)
├── app.py                           ← Streamlit web app (the UI)
├── heart_model.pkl                  ← Saved trained model + scaler
├── heart.csv                        ← Dataset used for training
├── requirements.txt                 ← All Python libraries needed
└── README.md                        ← This file
```

---

## 🤔 What Does This Project Do?

1. **Takes patient health info** as input (age, blood pressure, heart rate, etc.)
2. **Runs it through a trained ML model**
3. **Predicts** whether the patient has heart disease or not
4. **Shows a probability chart** (pie chart) of the risk

---

## 📊 Dataset

- **File:** `heart.csv`
- **Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Total columns used for training:** 11 (2 columns `chol` and `fbs` were dropped after correlation analysis)

| Column | What it means |
|---|---|
| `age` | Age of the patient |
| `sex` | Gender (0 = Female, 1 = Male) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure |
| `restecg` | Resting ECG result (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (0 = No, 1 = Yes) |
| `oldpeak` | ST depression after exercise |
| `slope` | Slope of the ST segment (0–2) |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type (1–3) |
| `target` | **Output** → 1 = Heart Disease, 0 = No Disease |

---

## 🧠 How the Model Was Built (Inside the Notebook)

### Step 1 — Load & Explore Data
- Loaded `heart.csv` using Pandas
- Checked shape, info, missing values, and duplicates
- Found **no missing values**, removed duplicate rows

### Step 2 — Feature Selection
- Plotted a **correlation heatmap** using Seaborn
- Dropped `chol` (cholesterol) and `fbs` (fasting blood sugar) — they had very low correlation with the target

### Step 3 — Train/Test Split
- Split data: **80% training, 20% testing**
- Used `random_state=42` for reproducibility

### Step 4 — Scaling
- Applied **StandardScaler** to normalize the feature values
- Fit the scaler **only on training data** (to avoid data leakage)

### Step 5 — Tried 5 Models
Compared these models before choosing the best:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier

Evaluated each using: **Accuracy, F1 Score, Precision, Recall, ROC AUC**

### Step 6 — Hyperparameter Tuning (GridSearchCV)
Fine-tuned the top 3 models:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Used **3-fold cross-validation** during tuning.

### Step 7 — Final Model Selected
**Logistic Regression** was chosen as the final model with these best params:
```
C=0.01, penalty='l2', solver='lbfgs'
```

### Step 8 — Save the Model
Saved both the **trained model + scaler** together in one file using `joblib`:
```python
model_data = {"model": final_model, "scaler": scaler}
joblib.dump(model_data, "heart_model.pkl")
```

---

## 🖥️ How to Run the App

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### Step 2 — Install the Required Libraries
```bash
pip install -r requirements.txt
```

### Step 3 — Make Sure `heart_model.pkl` is Present
The `.pkl` file must be in the same folder as `app.py`. If it's missing, run the notebook first to generate it.

### Step 4 — Run the App
```bash
streamlit run app.py
```

### Step 5 — Open in Browser
Streamlit will automatically open the app. If it doesn't, go to:
```
http://localhost:8501
```

---

## 📦 Requirements

Create a `requirements.txt` file with:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
seaborn
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🎯 How to Use the App

1. Open the app in your browser
2. Fill in the patient details on the left and right panels:
   - Age, Sex, Chest Pain Type, Blood Pressure, ECG, etc.
3. Click **"Run Assessment"**
4. The app will show:
   - ✅ **No Heart Disease** — in green
   - ⚠️ **Heart Disease Detected** — in red
   - A **pie chart** showing probability of each outcome

---

## 📈 Model Performance (After Tuning)

> Final model: **Logistic Regression**

| Metric | Training | Testing |
|---|---|---|
| Accuracy | ~85% | ~84% |
| F1 Score | ~85% | ~84% |
| Precision | ~85% | ~84% |
| Recall | ~86% | ~85% |
| ROC AUC | ~85% | ~84% |

*(Exact values may vary slightly based on your environment)*

---

## ⚠️ Disclaimer

> This app is for **educational and reference purposes only**.
> It is **not** a substitute for professional medical advice.
> Always consult a certified doctor or cardiologist for health decisions.

---
