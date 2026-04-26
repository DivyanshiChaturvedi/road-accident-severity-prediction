"""
Road Accident Severity Prediction Web Application
Streamlit app for predicting accident severity using multiple ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚦",
    layout="wide"
)

# Paths
DATA_PATH = "data/RTA Dataset.csv"
MODEL_DIR = "models"

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eafdfc 0%, #f8fafc 45%, #fff7e6 100%);
    }
    .main-hero {
        background: linear-gradient(135deg, #061826 0%, #0f766e 55%, #f59e0b 100%);
        padding: 36px;
        border-radius: 28px;
        color: white;
        box-shadow: 0 24px 70px rgba(15, 118, 110, 0.28);
        margin-bottom: 26px;
    }
    .main-title {
        font-size: 46px;
        font-weight: 900;
        line-height: 1.05;
        margin-bottom: 12px;
    }
    .main-subtitle {
        font-size: 18px;
        max-width: 950px;
        opacity: 0.95;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(15, 118, 110, 0.16);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
    }
    .prediction-card {
        background: white;
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        border-left: 12px solid #0f766e;
    }
    .fatal {
        border-left-color: #dc2626;
    }
    .serious {
        border-left-color: #f59e0b;
    }
    .slight {
        border-left-color: #16a34a;
    }
    .small-text {
        color: #64748b;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load the dataset"""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def load_models():
    """Load all trained models"""
    models = {}
    model_files = ['random_forest.pkl', 'gradient_boosting.pkl', 'logistic_regression.pkl', 
                   'decision_tree.pkl', 'knn.pkl', 'svm.pkl', 'adaboost.pkl']
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[model_file.replace('.pkl', '')] = joblib.load(model_path)
    
    return models


def load_encoders():
    """Load label encoders and target encoder"""
    encoders = {}
    if os.path.exists(os.path.join(MODEL_DIR, 'label_encoders.pkl')):
        with open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'rb') as f:
            encoders['label_encoders'] = joblib.load(f)
    if os.path.exists(os.path.join(MODEL_DIR, 'target_encoder.pkl')):
        with open(os.path.join(MODEL_DIR, 'target_encoder.pkl'), 'rb') as f:
            encoders['target_encoder'] = joblib.load(f)
    if os.path.exists(os.path.join(MODEL_DIR, 'scaler.pkl')):
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            encoders['scaler'] = joblib.load(f)
    return encoders


def train_all_models(X, y, model_info):
    """Train all classification models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
        'adaboost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        trained_models[name] = model
        
        # Save model
        joblib.dump(model, os.path.join(MODEL_DIR, f'{name}.pkl'))
    
    return results, trained_models, scaler, X_test_scaled, y_test


def severity_advice(prediction):
    """Get advice based on severity prediction"""
    text = prediction.lower()
    
    if "fatal" in text:
        return ("fatal", "Fatal Injury Risk", "High-risk accident condition detected. Immediate emergency response and strict preventive action are recommended.")
    if "serious" in text:
        return ("serious", "Serious Injury Risk", "Moderate to high accident severity risk. Emergency support should be prepared quickly.")
    return ("slight", "Slight Injury Risk", "Lower severity risk detected. Safety monitoring and cautious driving are still recommended.")


def main():
    """Main application: Dashboard, Input, Prediction"""
    st.markdown("""
    <div class="main-hero">
        <div class="main-title">🚦 Road Accident Severity Prediction Dashboard</div>
        <div class="main-subtitle">
            Enter accident conditions below to predict severity and get safety precautions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    if df is None:
        st.error("Dataset not found! Please run train_model.py first to generate the dataset.")
        st.info("Run: `python train_model.py`")
        return

    # Dashboard section
    st.subheader("📊 Accident Data Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        target_col = 'Accident_Severity'
        if target_col in df.columns:
            st.metric("Severity Classes", df[target_col].nunique())
        else:
            st.metric("Features", df.shape[1])
    with col4:
        st.metric("Target Column", "Accident_Severity")
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    if 'Accident_Severity' in df.columns:
        st.markdown("### Accident Severity Distribution")
        severity_counts = df["Accident_Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        fig = px.pie(
            severity_counts,
            names="Severity",
            values="Count",
            hole=0.45,
            title="Accident Severity Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("📝 Enter Accident Conditions for Prediction")
    # Input form
    models = load_models()
    encoders = load_encoders()
    label_encoders = encoders.get('label_encoders', {})
    target_encoder = encoders.get('target_encoder', LabelEncoder())
    scaler = encoders.get('scaler', StandardScaler())

    col1, col2 = st.columns(2)
    with col1:
        day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        age = st.number_input("Age of Driver", min_value=18, max_value=70, value=30)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        education = st.selectbox("Educational Level", ['Illiterate', 'Primary', 'Secondary', 'Tertiary'])
        vehicle = st.selectbox("Vehicle Type", ['Automobile', 'Motorcycle', 'Lorry', 'Bus', 'Taxi'])
    with col2:
        experience = st.number_input("Driving Experience (years)", min_value=0, max_value=30, value=5)
        speed_limit = st.selectbox("Speed Limit", [30, 40, 50, 60, 70, 80, 100])
        weather = st.selectbox("Weather", ['Clear', 'Cloudy', 'Rain', 'Fog'])
        light = st.selectbox("Road Light Conditions", ['Daylight', 'Night', 'Twilight'])
        surface = st.selectbox("Road Surface Conditions", ['Dry', 'Wet', 'Snow', 'Ice'])
    cause = st.selectbox("Cause of Accident", ['Overspeed', 'Drunk Driving', 'Distracted', 'Fatigue', 'Mechanical Failure', 'Weather'])

    input_data = pd.DataFrame({
        'Day_of_Week': [day_of_week],
        'Age_of_Driver': [age],
        'Sex': [sex],
        'Educational_Level': [education],
        'Vehicle_Driver': [vehicle],
        'Driving_Experience': [experience],
        'Speed_Limit': [speed_limit],
        'Weather': [weather],
        'Road_Light_Conditions': [light],
        'Road_Surface_Conditions': [surface],
        'Cause_of_Accident': [cause]
    })
    for column in input_data.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            input_data[column] = label_encoders[column].transform(input_data[column])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Severity", type="primary"):
        st.markdown("### Prediction Result")
        if not models:
            st.error("No trained models found. Please train models first.")
            return
        rf_model = models.get('random_forest', list(models.values())[0])
        pred = rf_model.predict(input_scaled)[0]
        pred_label = target_encoder.inverse_transform([pred])[0]
        severity, title, advice = severity_advice(pred_label)
        st.markdown(f"""
        <div class="prediction-card {severity}">
            <h3>{title}</h3>
            <p><b>Predicted Severity:</b> {pred_label}</p>
            <p><b>Precaution:</b> {advice}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Probability Distribution (Random Forest)")
        prob = rf_model.predict_proba(input_scaled)[0]
        prob_df = pd.DataFrame({
            'Class': target_encoder.classes_,
            'Probability': prob
        })
        fig = px.bar(
            prob_df,
            x='Class',
            y='Probability',
            color='Probability',
            color_continuous_scale="Teal",
            title="Prediction Probabilities"
        )
        st.plotly_chart(fig, use_container_width=True)


def home_page(df):
    """Home page with dashboard"""
    st.subheader("📊 Accident Data Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Total Features", df.shape[1])
    
    with col3:
        target_col = 'Accident_Severity'
        if target_col in df.columns:
            st.metric("Severity Classes", df[target_col].nunique())
        else:
            st.metric("Features", df.shape[1])
    
    with col4:
        st.metric("Target Column", "Accident_Severity")
    
    # Dataset preview
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Severity distribution
    if 'Accident_Severity' in df.columns:
        st.markdown("### Accident Severity Distribution")
        severity_counts = df["Accident_Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        
        fig = px.pie(
            severity_counts,
            names="Severity",
            values="Count",
            hole=0.45,
            title="Accident Severity Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)


def data_overview_page(df):
    """Data overview page"""
    st.header("📈 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
    
    with col2:
        st.markdown("#### Column Names")
        st.write(df.columns.tolist())
    
    st.markdown("### Data Types")
    st.dataframe(df.dtypes)
    
    st.markdown("### Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])
    
    st.markdown("### Sample Data")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Feature analysis
    st.markdown("### Feature Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Numeric Features")
        st.write(numeric_cols)
    
    with col2:
        st.markdown("#### Categorical Features")
        st.write(categorical_cols)


def model_comparison_page(df):
    """Model comparison page"""
    st.header("🤖 Model Comparison")
    
    target_col = 'Accident_Severity'
    
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset!")
        return
    
    # Prepare data
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Train models
    with st.spinner('Training all models... This may take a minute...'):
        results, trained_models, scaler, X_test, y_test = train_all_models(X, y_encoded, {})
    
    # Display results
    st.markdown("### Model Accuracy Comparison")
    
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': list(results.values())
    }).sort_values('Accuracy', ascending=False)
    
    results_df['Accuracy %'] = results_df['Accuracy'] * 100
    results_df['Accuracy %'] = results_df['Accuracy %'].round(2)
    
    # Bar chart
    fig = px.bar(
        results_df,
        x='Model',
        y='Accuracy %',
        color='Accuracy %',
        title="Model Accuracy Comparison",
        color_continuous_scale="Teal"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.dataframe(results_df, use_container_width=True)
    
    # Best model
    best_model = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Accuracy %']
    
    st.success(f"🏆 Best Model: {best_model} with {best_accuracy}% accuracy")
    
    # Detailed comparison
    st.markdown("### Detailed Model Performance")
    
    # Train-test split for detailed metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    scaler_detailed = StandardScaler()
    X_train_scaled = scaler_detailed.fit_transform(X_train)
    X_test_scaled = scaler_detailed.transform(X_test)
    
    for name, model in trained_models.items():
        with st.expander(f"📊 {name.replace('_', ' ').title()}"):
            y_pred = model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
            st.json(report)


def prediction_page(df):
    """Prediction page"""
    st.header("🎯 Make Prediction")
    
    target_col = 'Accident_Severity'
    
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found!")
        return
    
    # Load models and encoders
    models = load_models()
    encoders = load_encoders()
    
    if not models:
        st.warning("No trained models found. Training new models...")
        
        # Prepare data
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Encode
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
        
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        # Train
        results, trained_models, scaler, X_test, y_test = train_all_models(X, y_encoded, {})
        models = trained_models
    else:
        # Load encoders from saved files
        label_encoders = encoders.get('label_encoders', {})
        target_encoder = encoders.get('target_encoder', LabelEncoder())
        scaler = encoders.get('scaler', StandardScaler())
    
    # Input form
    st.markdown("### Enter Accident Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        age = st.number_input("Age of Driver", min_value=18, max_value=70, value=30)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        education = st.selectbox("Educational Level", ['Illiterate', 'Primary', 'Secondary', 'Tertiary'])
        vehicle = st.selectbox("Vehicle Type", ['Automobile', 'Motorcycle', 'Lorry', 'Bus', 'Taxi'])
    
    with col2:
        experience = st.number_input("Driving Experience (years)", min_value=0, max_value=30, value=5)
        speed_limit = st.selectbox("Speed Limit", [30, 40, 50, 60, 70, 80, 100])
        weather = st.selectbox("Weather", ['Clear', 'Cloudy', 'Rain', 'Fog'])
        light = st.selectbox("Road Light Conditions", ['Daylight', 'Night', 'Twilight'])
        surface = st.selectbox("Road Surface Conditions", ['Dry', 'Wet', 'Snow', 'Ice'])
    
    cause = st.selectbox("Cause of Accident", ['Overspeed', 'Drunk Driving', 'Distracted', 'Fatigue', 'Mechanical Failure', 'Weather'])
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Day_of_Week': [day_of_week],
        'Age_of_Driver': [age],
        'Sex': [sex],
        'Educational_Level': [education],
        'Vehicle_Driver': [vehicle],
        'Driving_Experience': [experience],
        'Speed_Limit': [speed_limit],
        'Weather': [weather],
        'Road_Light_Conditions': [light],
        'Road_Surface_Conditions': [surface],
        'Cause_of_Accident': [cause]
    })
    
    # Encode input using saved encoders
    for column in input_data.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            input_data[column] = label_encoders[column].transform(input_data[column])
    
    # Scale
    input_scaled = scaler.transform(input_data)
    
    # Predict
    if st.button("Predict Severity", type="primary"):
        st.markdown("### Prediction Results")
        
        results = {}
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
            # Convert numeric prediction back to class label
            pred_label = target_encoder.inverse_transform([pred])[0]
            results[name] = {
                'prediction': pred_label,
                'probabilities': prob
            }
        
        # Display predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Predictions")
            for name, result in results.items():
                st.write(f"**{name.replace('_', ' ').title()}**: {result['prediction']}")
        
        with col2:
            st.markdown("#### Best Model Prediction")
            # Use random forest as default best
            rf_result = results.get('random_forest', results.get(list(results.keys())[0]))
            severity_class = rf_result['prediction']
            severity, title, advice = severity_advice(severity_class)
            
            st.markdown(f"""
            <div class="prediction-card {severity}">
                <h3>{title}</h3>
                <p>{advice}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability chart
        st.markdown("### Probability Distribution (Random Forest)")
        prob_df = pd.DataFrame({
            'Class': target_encoder.classes_,
            'Probability': rf_result['probabilities']
        })
        
        fig = px.bar(
            prob_df,
            x='Class',
            y='Probability',
            color='Probability',
            color_continuous_scale="Teal",
            title="Prediction Probabilities"
        )
        st.plotly_chart(fig, use_container_width=True)




if __name__ == "__main__":
    main()