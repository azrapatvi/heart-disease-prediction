# 🫀 Heart Disease Prediction App

A machine learning web app that predicts whether a person has heart disease — using **two models**: Logistic Regression and a Deep Learning ANN. Built with **Python**, **Scikit-learn**, **TensorFlow/Keras**, and **Streamlit**.


---

## 🗂️ Project Structure

```
heart-disease-prediction/
│
├── heart_disease_prediction.ipynb   ← Jupyter notebook (EDA + model training)
├── ann_heart_disease_prediction.ipynb               ← ANN model training notebook
├── main.py                           ← Streamlit web app (UI)
├── heart_model.pkl                  ← Saved Logistic Regression model + scaler
├── ann_model.h5                     ← Saved ANN (Keras) model 
├── requirements.txt                 ← All libraries needed
└── README.md                        ← This file
```

---

## 🤔 What Does This App Do?

1. User fills in **patient health details** (age, blood pressure, heart rate, etc.)
2. User **selects a model** — Logistic Regression or ANN
3. App **predicts** if the patient has heart disease or not
4. App shows a **probability pie chart** of the risk

---

## 📊 Dataset

- **File:** `heart.csv`
- **Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Columns dropped:** `chol` and `fbs` — removed after correlation analysis (low correlation with target)
- **Final features used for training:** 11 columns

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

## 🧠 How the Models Were Built

### ── Common Steps (done for both models) ──

**Step 1 — Load & Explore Data**
- Loaded `heart.csv` using Pandas
- Checked shape, info, missing values, duplicates
- No missing values found; duplicate rows removed

**Step 2 — Feature Selection**
- Plotted a correlation heatmap using Seaborn
- Dropped `chol` and `fbs` — very low correlation with target

**Step 3 — Train/Test Split**
- 80% training, 20% testing
- `random_state=42` for reproducibility

**Step 4 — Scaling**
- Used `StandardScaler` to normalize features
- Scaler fitted **only on training data** (prevents data leakage)

---

### 🔵 Model 1 — Logistic Regression

**Step 5 — Tried 5 Models**

Compared all of these before picking the best:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier

Each model was evaluated using: Accuracy, F1 Score, Precision, Recall, ROC AUC

**Step 6 — Hyperparameter Tuning (GridSearchCV)**

Top 3 models were fine-tuned using GridSearchCV with 3-fold cross-validation:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

**Step 7 — Final Model**

**Logistic Regression** was selected as the final model with best params:
```
C=0.01, penalty='l2', solver='lbfgs'
```

**Step 8 — Save**
```python
model_data = {"model": final_model, "scaler": scaler}
joblib.dump(model_data, "heart_model.pkl")
```

---

### 🟢 Model 2 — ANN (Artificial Neural Network)

**Step 5 — ANN Architecture**

Built using TensorFlow/Keras with the following layers:

| Layer | Neurons | Activation | Dropout |
|---|---|---|---|
| Input | 11 features | — | — |
| Dense 1 | 128 | ReLU | 0.3 |
| Dense 2 | 64 | ReLU | 0.2 |
| Dense 3 | 32 | ReLU | 0.2 |
| Dense 4 | 16 | ReLU | 0.1 |
| Output | 1 | Sigmoid | — |

> **Dropout** layers are added after each Dense layer to prevent overfitting — they randomly turn off some neurons during training so the model doesn't memorize the data.

**Step 6 — Compile**
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Step 7 — Training with Early Stopping**
```python
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
          epochs=150, callbacks=[early_stopping])
```

> **Early Stopping** means: if the model stops improving for 5 epochs in a row, training stops automatically and the best weights are restored. This saves time and prevents overfitting.

**Step 8 — Prediction**
```python
y_pred = model.predict(X_test_scaled)
y_pred_class = (y_pred > 0.5).astype(int)   # threshold: >0.5 = Disease
```

**Step 9 — Save**
```python
model.save("ann_model.h5")
```

---

## 📦 Requirements

Add this to your `requirements.txt`:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
seaborn
tensorflow
```


---

## 🎯 How to Use the App

1. Open the app in your browser
2. **Select a model** from the dropdown at the top:
   - 🔵 Logistic Regression — simple and fast
   - 🟢 ANN — deep learning, more complex
3. Fill in the patient details (age, sex, chest pain, blood pressure, etc.)
4. Click **"Run Assessment"**
5. The app shows:
   - ✅ **No Heart Disease** — in green
   - ⚠️ **Heart Disease Detected** — in red
   - Which model was used for the prediction
   - A **pie chart** showing the probability breakdown

---

## 📈 Model Performance

### 🔵 Logistic Regression (after tuning)

| Metric | Training | Testing |
|---|---|---|
| Accuracy | ~85% | ~84% |
| F1 Score | ~85% | ~84% |
| Precision | ~85% | ~84% |
| Recall | ~86% | ~85% |
| ROC AUC | ~85% | ~84% |

### 🟢 ANN (Neural Network)

| Metric | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Total Epochs Run | 13 (Early Stopping triggered) |
| Best Epoch | 8 |
| Training Accuracy (Best Epoch) | ~83.06% |
| Validation Accuracy | ~88.52% |
| Best Validation Loss | 0.3549 |
| Threshold | 0.5 (above = Heart Disease) |

> Early Stopping triggered at Epoch 13 (patience=5). Best weights were restored from **Epoch 8**, which achieved the lowest validation loss of **0.3549** and a validation accuracy of **88.52%**.

---

## ⚠️ Disclaimer

> This app is for **educational and reference purposes only**.
> It is **not** a substitute for professional medical advice.
> Always consult a certified doctor or cardiologist for health decisions.
