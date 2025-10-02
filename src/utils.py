"""
Utility functions for the Streamlit Employee Performance Prediction App
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def load_models_and_data():
    """
    Load the trained model, scaler, and encoders.
    Handles different possible paths for the model files.
    """
    # Possible model file locations
    possible_paths = [
        Path("../data"),  # Parent directory data folder (most likely)
        get_project_root().parent / "data",
        get_project_root() / "data",
        Path("data"),
        Path(r"c:\Users\tonyn\Downloads\IABAC exam elizabeth\data"),
        Path(r"D:\Data Science Classes\Machine Learning\Group Assignment\IABAC Exams\Employee Performance Analysis\data"),
    ]
    
    model_files = {
        'model': 'best_model.pkl',
        'scaler': 'scaler.pkl', 
        'encoders': 'all_encoders.pkl'
    }
    
    loaded_objects = {}
    
    for path in possible_paths:
        if path.exists():
            try:
                all_files_exist = all((path / filename).exists() for filename in model_files.values())
                if all_files_exist:
                    for key, filename in model_files.items():
                        loaded_objects[key] = joblib.load(path / filename)
                    
                    st.success(f"✅ Models loaded successfully from: {path}")
                    return loaded_objects['model'], loaded_objects['scaler'], loaded_objects['encoders']
            except Exception as e:
                continue
    
    # If we reach here, no valid model files were found
    st.error("❌ Model files not found. Please ensure the model has been trained and saved.")
    st.info("Expected files: best_model.pkl, scaler.pkl, all_encoders.pkl")
    return None, None, None

def create_sample_data():
    """Create sample data for testing if no models are available."""
    return {
        'Age': 35,
        'TotalWorkExperienceInYears': 10,
        'YearsInCurrentRole': 4,
        'YearsWithCurrManager': 3,
        'YearsSinceLastPromotion': 2,
        'EmpDepartment': 'Development',
        'EmpJobRole': 'Developer',
        'EmpEnvironmentSatisfaction': 3,
        'EmpWorkLifeBalance': 3,
        'EmpLastSalaryHikePercent': 15
    }

def validate_input_data(input_dict):
    """Validate user input data."""
    errors = []
    
    # Check numeric ranges
    if input_dict['Age'] < 18 or input_dict['Age'] > 65:
        errors.append("Age must be between 18 and 65")
    
    if input_dict['TotalWorkExperienceInYears'] < 0 or input_dict['TotalWorkExperienceInYears'] > 40:
        errors.append("Total work experience must be between 0 and 40 years")
    
    if input_dict['YearsInCurrentRole'] > input_dict['TotalWorkExperienceInYears']:
        errors.append("Years in current role cannot exceed total work experience")
    
    if input_dict['EmpLastSalaryHikePercent'] < 10 or input_dict['EmpLastSalaryHikePercent'] > 25:
        errors.append("Salary hike percentage must be between 10% and 25%")
    
    return errors

def format_prediction_confidence(prediction_proba):
    """Format prediction probabilities for display."""
    confidence_data = {}
    performance_levels = ['Good', 'Excellent', 'Outstanding']
    
    for i, level in enumerate(performance_levels):
        if i < len(prediction_proba):
            confidence_data[level] = prediction_proba[i] * 100
        else:
            confidence_data[level] = 0.0
    
    return confidence_data

def get_performance_color(performance_rating):
    """Get color code for performance rating."""
    colors = {
        'Good': '#ff6b6b',
        'Excellent': '#4ecdc4', 
        'Outstanding': '#45b7d1'
    }
    return colors.get(performance_rating, '#95a5a6')

def get_department_info():
    """Get department performance information."""
    return {
        'Data Science': {'avg_rating': 3.05, 'description': 'High-performing, specialized team'},
        'Development': {'avg_rating': 3.09, 'description': 'Strongest department overall'},
        'Finance': {'avg_rating': 2.78, 'description': 'Needs performance improvement'},
        'Human Resources': {'avg_rating': 2.93, 'description': 'Stable, consistent performance'},
        'Research & Development': {'avg_rating': 2.92, 'description': 'Solid performance with potential'},
        'Sales': {'avg_rating': 2.86, 'description': 'Mixed performance, room for growth'}
    }

def get_feature_importance_data():
    """Get feature importance data for visualization."""
    return {
        'Environment Satisfaction': 0.28,
        'Salary Hike Percentage': 0.22,
        'Years Since Promotion': 0.18,
        'Department': 0.12,
        'Years in Current Role': 0.08,
        'Job Role': 0.06,
        'Years with Manager': 0.04,
        'Total Experience': 0.02
    }
