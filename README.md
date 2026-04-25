cat > README.md << 'EOF'
# Road Accident Severity Prediction Dashboard

This project predicts road accident severity using machine learning and the Road Accident Severity Dataset.

## Objective

The objective of this project is to predict accident severity based on driver, vehicle, weather, road, and casualty-related features.

## Dataset

Dataset used:

Road Accident Severity Dataset

Target column:

Accident_severity

Classes:

- Slight Injury
- Serious Injury
- Fatal injury

## Features

- Accident data dashboard
- Severity distribution chart
- Weather condition analysis
- Light condition analysis
- Cause of accident analysis
- Random Forest machine learning model
- Model accuracy
- Classification report
- Confusion matrix
- Interactive severity prediction

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib

## How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
