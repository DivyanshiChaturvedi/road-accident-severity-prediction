# Road Accident Severity Prediction

A production-ready ML classification pipeline that predicts road accident severity using the US Accidents dataset. Built to demonstrate the difference between a model that fits and a model that generalizes.

---

## 🎯 Problem Statement

Given driver, vehicle, weather, road, and casualty-related features, predict accident severity (Slight Injury / Serious Injury / Fatal Injury) with production-grade reliability.

---

## 🏗️ Engineering Decisions

- **Feature Engineering:** Built 12 features from raw data (weather conditions, road type, driver demographics, vehicle type, time-of-day) — applied systematic feature selection to reduce dimensionality without accuracy loss
- **Model Selection:** Trained and evaluated 7 classifiers; selected Random Forest based on generalization to held-out test data, not raw training score
- **Production Design:** Severity scoring outputs color-coded alerts translating ML predictions into operationally actionable signals

---

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| ✅ **Random Forest** (selected) | **98.25%** |
| Gradient Boosting | 97.75% |
| Decision Tree | 94.50% |

> Selection criterion: generalization on held-out test set, not training accuracy.

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** Scikit-learn, XGBoost, Pandas, NumPy
- **Interface:** Streamlit, Plotly
- **Model Persistence:** Joblib

---

## 🚀 How to Run

```bash
git clone https://github.com/DivyanshiChaturvedi/road-accident-severity-prediction
cd road-accident-severity-prediction
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

---

## 📁 Project Structure

```
road-accident-severity-prediction/
├── data/              # Dataset
├── models/            # Saved trained models
├── templates/         # UI templates
├── train_model.py     # Model training + evaluation
├── app.py             # Streamlit dashboard
└── requirements.txt
```
