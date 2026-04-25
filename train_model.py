"""
Road Accident Severity Prediction Model Training Script
This script trains multiple machine learning models to predict accident severity.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Sample data generation for demonstration
def generate_sample_data():
    """Generate sample road accident data"""
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'Day_of_Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                         'Friday', 'Saturday', 'Sunday'], n_samples),
        'Age_of_Driver': np.random.randint(18, 70, n_samples),
        'Sex': np.random.choice(['Male', 'Female'], n_samples),
        'Educational_Level': np.random.choice(['Primary', 'Secondary', 'Tertiary', 
                                               'Illiterate'], n_samples),
        'Vehicle_Driver': np.random.choice(['Automobile', 'Motorcycle', 'Lorry', 
                                            'Bus', 'Taxi'], n_samples),
        'Driving_Experience': np.random.randint(0, 30, n_samples),
        'Speed_Limit': np.random.choice([30, 40, 50, 60, 70, 80, 100], n_samples),
        'Weather': np.random.choice(['Clear', 'Rain', 'Fog', 'Cloudy'], n_samples),
        'Road_Light_Conditions': np.random.choice(['Daylight', 'Night', 'Twilight'], n_samples),
        'Road_Surface_Conditions': np.random.choice(['Dry', 'Wet', 'Snow', 'Ice'], n_samples),
        'Cause_of_Accident': np.random.choice(['Overspeed', 'Drunk Driving', 'Distracted', 
                                               'Fatigue', 'Mechanical Failure', 'Weather'], n_samples),
    }
    
    # Generate severity based on risk factors
    severity = []
    for i in range(n_samples):
        risk = 0
        if data['Speed_Limit'][i] > 60:
            risk += 2
        if data['Driving_Experience'][i] < 3:
            risk += 2
        if data['Weather'][i] in ['Rain', 'Fog']:
            risk += 1
        if data['Road_Surface_Conditions'][i] in ['Wet', 'Ice', 'Snow']:
            risk += 1
        if data['Age_of_Driver'][i] < 25:
            risk += 1
        if data['Cause_of_Accident'][i] in ['Overspeed', 'Drunk Driving']:
            risk += 2
            
        # Assign severity based on risk score
        if risk >= 6:
            severity.append('Fatal Injury')
        elif risk >= 3:
            severity.append('Serious Injury')
        else:
            severity.append('Slight Injury')
    
    data['Accident_Severity'] = severity
    return pd.DataFrame(data)


def train_all_models():
    """Train multiple accident severity prediction models"""
    print("=" * 60)
    print("Road Accident Severity Prediction - Model Training")
    print("=" * 60)
    
    # Generate and save data
    print("\n[1/4] Generating sample data...")
    df = generate_sample_data()
    
    # Save the dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'RTA Dataset.csv')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Dataset saved to {data_path}")
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features and target
    target_col = 'Accident_Severity'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode categorical variables
    print("\n[2/4] Encoding categorical variables...")
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save encoders and scaler
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
    joblib.dump(target_encoder, os.path.join(model_dir, 'target_encoder.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print("Encoders and scaler saved.")
    
    # Define models
    print("\n[3/4] Training multiple models...")
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
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Save model
        model_path = os.path.join(model_dir, f'{name}.pkl')
        joblib.dump(model, model_path)
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Summary
    print("\n[4/4] Model Training Complete!")
    print("=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, accuracy) in enumerate(sorted_results, 1):
        print(f"{rank}. {name.replace('_', ' ').title()}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Best model
    best_model = sorted_results[0][0]
    best_accuracy = sorted_results[0][1]
    print(f"\n🏆 Best Model: {best_model.replace('_', ' ').title()} with {best_accuracy*100:.2f}% accuracy")
    
    # Save summary
    summary = {
        'best_model': best_model,
        'best_accuracy': best_accuracy,
        'all_results': results,
        'target_classes': list(target_encoder.classes_)
    }
    joblib.dump(summary, os.path.join(model_dir, 'model_summary.pkl'))
    
    print(f"\nAll models saved to: {model_dir}")
    print("\nTo run the web app, use: streamlit run app.py")
    
    return results


if __name__ == '__main__':
    train_all_models()