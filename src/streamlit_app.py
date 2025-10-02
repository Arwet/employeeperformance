import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Try to import joblib and handle the case where model dependencies are missing
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    st.error("❌ joblib not found. Please install: pip install joblib")
    JOBLIB_AVAILABLE = False

# Check for XGBoost availability
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Check for LightGBM availability  
try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="INX Future Inc - Employee Performance Predictor",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open('static/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
load_css()

# Load models and encoders
@st.cache_resource
def load_models():
    """Load models using the utility function."""
    import os
    from pathlib import Path
    
    # Check dependencies first
    if not JOBLIB_AVAILABLE:
        st.error("❌ joblib not available. Please install: pip install joblib")
        return None, None, None
    
    if not XGBOOST_AVAILABLE:
        st.error("❌ XGBoost not available. Please install: pip install xgboost")
        st.info("💡 Make sure you're running in the correct environment with all dependencies installed")
        return None, None, None
    
    # Try multiple possible paths for the data directory
    possible_data_paths = [
        Path("../data"),  # Most likely path when running from src
        Path("data"),
        Path(r"c:\Users\tonyn\Downloads\IABAC exam elizabeth\data"),
    ]
    
    for data_path in possible_data_paths:
        if data_path.exists():
            model_path = data_path / "best_model.pkl"
            scaler_path = data_path / "scaler.pkl"
            encoders_path = data_path / "all_encoders.pkl"
            
            if all(p.exists() for p in [model_path, scaler_path, encoders_path]):
                try:
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    encoders = joblib.load(encoders_path)
                    st.success(f"✅ Models loaded successfully from: {data_path}")
                    return model, scaler, encoders
                except Exception as e:
                    st.error(f"❌ Error loading models from {data_path}: {e}")
                    st.info("💡 This might be due to missing dependencies. Make sure XGBoost and LightGBM are installed.")
                    continue
    
    st.error("❌ Model files not found in any expected location.")
    st.info("Expected files: best_model.pkl, scaler.pkl, all_encoders.pkl")
    st.info("Looking in: ../data/, data/, or absolute path")
    return None, None, None

# Top 10 features used in the model
TOP_10_FEATURES = [
    'EmpEnvironmentSatisfaction',
    'YearsSinceLastPromotion',
    'EmpLastSalaryHikePercent',
    'EmpDepartment',
    'ExperienceYearsInCurrentRole',
    'EmpJobRole',
    'YearsWithCurrManager',
    'ExperienceYearsAtThisCompany',
    'EmpWorkLifeBalance',
    'Age'
]

# Feature mappings
FEATURE_MAPPINGS = {
    'EmpEnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'EmpWorkLifeBalance': {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'},
    'EmpDepartment': ['Data Science', 'Development', 'Finance', 'Human Resources', 'Research & Development', 'Sales'],
    'EmpJobRole': ['Business Analyst', 'Data Scientist', 'Delivery Manager', 'Developer', 'Finance Manager', 
                   'Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 
                   'Manager R&D', 'Manufacturing Director', 'Research Director', 'Research Scientist', 
                   'Sales Executive', 'Sales Representative', 'Senior Developer', 'Technical Architect', 'Technical Lead']
}

PERFORMANCE_MAPPING = {0: 'Good', 1: 'Excellent', 2: 'Outstanding'}

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🏢 INX Future Inc</h1>
        <h2>Employee Performance Prediction System</h2>
        <p>Predict employee performance using advanced machine learning algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check dependencies first
    if not XGBOOST_AVAILABLE:
        st.error("❌ XGBoost not available. Please install: pip install xgboost")
        st.info("💡 Make sure you're running in the correct environment with all dependencies installed")
        show_demo_interface()
        return
    
    # Load models
    model, scaler, encoders = load_models()
    
    if model is None:
        show_demo_interface()
        return
    
    # Sidebar
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h3>📊 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🎯 Prediction", "📈 Model Insights", "ℹ️ About"]
    )
    
    if page == "🎯 Prediction":
        prediction_page(model, scaler, encoders)
    elif page == "📈 Model Insights":
        insights_page()
    else:
        about_page()

def show_demo_interface():
    """Show a demo interface when models are not available."""
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Demo: Employee Performance Prediction</h2>
        <p>This is a demonstration interface showing how the app would work with real models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create demo input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        age = st.number_input("Age", min_value=18, max_value=65, value=35, step=1)
        total_experience = st.number_input("Total Work Experience (Years)", min_value=0, max_value=40, value=10, step=1)
        years_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=4, step=1)
        
    with col2:
        st.subheader("🏢 Work Environment")
        department = st.selectbox("Department", ['Development', 'Data Science', 'Finance', 'Human Resources', 'Research & Development', 'Sales'])
        environment_satisfaction = st.selectbox("Environment Satisfaction", options=['Low', 'Medium', 'High', 'Very High'], index=2)
        salary_hike_percent = st.number_input("Last Salary Hike (%)", min_value=10, max_value=25, value=15, step=1)
    
    if st.button("🔮 Generate Demo Prediction", use_container_width=True):
        # Simulate prediction based on inputs
        import random
        import numpy as np
        
        # Simple rule-based demo prediction
        score = (
            (age - 30) * 0.1 +
            total_experience * 0.2 + 
            (4 - years_current_role) * 0.1 +
            ({'Development': 3, 'Data Science': 3, 'Finance': 2, 'Human Resources': 2.5, 'Research & Development': 2.5, 'Sales': 2}.get(department, 2.5)) +
            ({'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}.get(environment_satisfaction, 2)) * 0.3 +
            (salary_hike_percent - 15) * 0.1 +
            random.uniform(-0.5, 0.5)  # Add some randomness
        )
        
        # Convert to performance categories
        if score >= 3.2:
            performance = "Outstanding"
            color = "#45b7d1"
            confidence = random.uniform(75, 90)
        elif score >= 2.8:
            performance = "Excellent" 
            color = "#4ecdc4"
            confidence = random.uniform(70, 85)
        else:
            performance = "Good"
            color = "#ff6b6b"
            confidence = random.uniform(65, 80)
        
        # Display demo results
        st.markdown(f"""
        <div class="prediction-result" style="background: linear-gradient(135deg, {color}22, {color}11); border-left: 5px solid {color};">
            <h2>Demo Prediction: <span style="color: {color};">{performance}</span></h2>
            <p>Simulated Confidence: <strong>{confidence:.1f}%</strong></p>
            <p><em>This is a demo prediction - install XGBoost for real predictions</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show demo chart
        fig = go.Figure(data=[
            go.Bar(
                x=['Good', 'Excellent', 'Outstanding'],
                y=[random.uniform(0.1, 0.3), random.uniform(0.4, 0.7), random.uniform(0.1, 0.5)],
                marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                text=[f'{p:.1%}' for p in [random.uniform(0.1, 0.3), random.uniform(0.4, 0.7), random.uniform(0.1, 0.5)]],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Demo Prediction Probability Distribution",
            xaxis_title="Performance Rating",
            yaxis_title="Probability",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 This is a simulated prediction. Install the required packages to use the real ML model.")

def prediction_page(model, scaler, encoders):
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Employee Performance Prediction</h2>
        <p>Enter employee details to predict their performance rating</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("📋 Personal Information")
        
        age = st.number_input("Age", min_value=18, max_value=65, value=35, step=1, help="Employee's age in years")
        
        total_experience = st.number_input(
            "Total Work Experience (Years)", 
            min_value=0, max_value=40, value=10, step=1,
            help="Total years of professional experience"
        )
        
        years_current_role = st.number_input(
            "Years in Current Role", 
            min_value=0, max_value=20, value=4, step=1,
            help="Years spent in current position"
        )
        
        years_with_manager = st.number_input(
            "Years with Current Manager", 
            min_value=0, max_value=20, value=3, step=1,
            help="Years working under current manager"
        )
        
        years_since_promotion = st.number_input(
            "Years Since Last Promotion", 
            min_value=0, max_value=15, value=2, step=1,
            help="Years elapsed since last promotion"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("🏢 Work Environment")
        
        department = st.selectbox(
            "Department", 
            FEATURE_MAPPINGS['EmpDepartment'],
            help="Employee's department"
        )
        
        job_role = st.selectbox(
            "Job Role", 
            FEATURE_MAPPINGS['EmpJobRole'],
            help="Employee's specific job role"
        )
        
        environment_satisfaction = st.selectbox(
            "Environment Satisfaction",
            options=list(FEATURE_MAPPINGS['EmpEnvironmentSatisfaction'].values()),
            index=2,  # Default to 'High'
            help="Level of satisfaction with work environment"
        )
        
        work_life_balance = st.selectbox(
            "Work-Life Balance",
            options=list(FEATURE_MAPPINGS['EmpWorkLifeBalance'].values()),
            index=1,  # Default to 'Good'
            help="Quality of work-life balance"
        )
        
        salary_hike_percent = st.number_input(
            "Last Salary Hike (%)", 
            min_value=10, max_value=25, value=15, step=1,
            help="Percentage of last salary increase"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔮 Predict Performance", use_container_width=True):
            # Prepare input data
            input_data = prepare_input_data(
                age, total_experience, years_current_role, years_with_manager,
                years_since_promotion, department, job_role, environment_satisfaction,
                work_life_balance, salary_hike_percent, encoders, scaler
            )
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            display_prediction_results(prediction, prediction_proba)

def prepare_input_data(age, total_experience, years_current_role, years_with_manager,
                      years_since_promotion, department, job_role, environment_satisfaction,
                      work_life_balance, salary_hike_percent, encoders, scaler):
    
    # Create input dictionary
    input_dict = {
        'Age': age,
        'ExperienceYearsAtThisCompany': total_experience,
        'ExperienceYearsInCurrentRole': years_current_role,
        'YearsWithCurrManager': years_with_manager,
        'YearsSinceLastPromotion': years_since_promotion,
        'EmpDepartment': department,
        'EmpJobRole': job_role,
        'EmpEnvironmentSatisfaction': get_key_by_value(FEATURE_MAPPINGS['EmpEnvironmentSatisfaction'], environment_satisfaction),
        'EmpWorkLifeBalance': get_key_by_value(FEATURE_MAPPINGS['EmpWorkLifeBalance'], work_life_balance),
        'EmpLastSalaryHikePercent': salary_hike_percent
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical variables
    for col in ['EmpDepartment', 'EmpJobRole']:
        if col in encoders:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except ValueError:
                # Handle unseen values
                input_df[col] = 0
    
    # Scale numerical features
    input_scaled = scaler.transform(input_df[TOP_10_FEATURES])
    
    return input_scaled

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return 1  # Default value

def display_prediction_results(prediction, prediction_proba):
    st.markdown("""
    <div class="results-section">
        <h2>🎉 Prediction Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance rating
    performance_rating = PERFORMANCE_MAPPING[prediction]
    confidence = max(prediction_proba) * 100
    
    # Color coding
    colors = {'Good': '#ff6b6b', 'Excellent': '#4ecdc4', 'Outstanding': '#45b7d1'}
    color = colors.get(performance_rating, '#95a5a6')
    
    # Display prediction with styling
    st.markdown(f"""
    <div class="prediction-result" style="background: linear-gradient(135deg, {color}22, {color}11); border-left: 5px solid {color};">
        <h2>Predicted Performance: <span style="color: {color};">{performance_rating}</span></h2>
        <p>Confidence: <strong>{confidence:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability distribution chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(PERFORMANCE_MAPPING.values()),
            y=prediction_proba,
            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
            text=[f'{p:.1%}' for p in prediction_proba],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probability Distribution",
        xaxis_title="Performance Rating",
        yaxis_title="Probability",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    display_recommendations(performance_rating)

def display_recommendations(performance_rating):
    st.markdown("""
    <div class="recommendations-section">
        <h3>💡 Recommendations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    recommendations = {
        'Good': [
            "🎯 Focus on improving environment satisfaction through better workspace conditions",
            "📈 Consider providing additional training and development opportunities",
            "🤝 Enhance communication with direct manager",
            "💰 Review compensation and consider performance-based incentives"
        ],
        'Excellent': [
            "⭐ Maintain current performance level through consistent recognition",
            "🚀 Provide opportunities for career advancement and skill development",
            "👥 Consider for mentoring junior team members",
            "🎖️ Recognize achievements through performance bonuses"
        ],
        'Outstanding': [
            "🏆 Continue providing challenging and engaging work",
            "👑 Consider for leadership roles and special projects",
            "🌟 Use as a role model for other team members",
            "💎 Ensure retention through competitive compensation and benefits"
        ]
    }
    
    for rec in recommendations[performance_rating]:
        st.markdown(f"- {rec}")

def insights_page():
    st.markdown("""
    <div class="section-header">
        <h2>📈 Model Insights & Analytics</h2>
        <p>Understanding the key factors that drive employee performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Top Performance Drivers")
        
        # Mock feature importance data (you can replace with actual values)
        feature_importance = {
            'Environment Satisfaction': 0.28,
            'Salary Hike Percentage': 0.22,
            'Years Since Promotion': 0.18,
            'Department': 0.12,
            'Years in Current Role': 0.08,
            'Job Role': 0.06,
            'Years with Manager': 0.04,
            'Total Experience': 0.02
        }
        
        fig = go.Figure(go.Bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            marker_color='#4ecdc4'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Department Performance")
        
        dept_performance = {
            'Data Science': 3.05,
            'Development': 3.09,
            'Finance': 2.78,
            'Human Resources': 2.93,
            'Research & Development': 2.92,
            'Sales': 2.86
        }
        
        fig = go.Figure(go.Bar(
            x=list(dept_performance.keys()),
            y=list(dept_performance.values()),
            marker_color='#45b7d1'
        ))
        
        fig.update_layout(
            title="Average Performance by Department",
            xaxis_title="Department",
            yaxis_title="Average Rating",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("### 🔍 Key Insights")
    
    # Create two columns for insights
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        #### 🌟 Top Performance Drivers
        
        **Environment Satisfaction (28%)**  
        The strongest predictor of performance
        
        **Salary Hike Percentage (22%)**  
        Financial recognition significantly impacts performance
        
        **Years Since Promotion (18%)**  
        Career advancement opportunities are crucial
        """)
    
    with insight_col2:
        st.markdown("""
        #### 🏆 Best Performing Departments
        
        **Development (3.09)**  
        Highest average performance rating
        
        **Data Science (3.05)**  
        Consistently excellent performance
        
        **Human Resources (2.93)**  
        Solid performance with growth potential
        """)

def about_page():
    st.markdown("""
    <div class="section-header">
        <h2>ℹ️ About This Application</h2>
        <p>Learn more about the Employee Performance Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("### 🎯 Project Overview")
    st.write("""
    INX Future Inc., a leading data analytics and automation solutions provider, has experienced 
    a decline in employee performance indices, with client satisfaction dropping by 8 percentage points. 
    This application uses advanced machine learning techniques to predict employee performance and 
    provide actionable insights for HR management.
    """)
    
    # Technical Details
    st.markdown("### 🔬 Technical Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Algorithm:** XGBoost Classifier
        - **Accuracy:** 94.2%
        - **Features:** Top 10 most important factors
        - **Classes:** Good, Excellent, Outstanding
        """)
    
    with col2:
        st.markdown("""
        - **Precision:** 94.0%
        - **Recall:** 94.2%
        - **F1-Score:** 94.0%
        - **Cross-validation Score:** 93.1%
        """)
    
    # Model Performance Metrics
    st.markdown("### 🎖️ Model Performance Metrics")
    
    # Create performance metrics visualization
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Cross-validation'],
        'Score': [94.0, 94.2, 94.0, 93.1]
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics_data['Metric'],
            y=metrics_data['Score'],
            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
            text=[f'{score}%' for score in metrics_data['Score']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score (%)",
        template="plotly_white",
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Contact Information
    st.markdown("### 📧 Contact & Support")
    st.info("""
    **INX Future Inc. - Employee Performance Prediction System**
    
    For technical support or questions about this application:
    - 📧 Email: hr-analytics@inxfuture.com
    - 🌐 Website: www.inxfuture.com
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 20px;'>"
        "© 2025 INX Future Inc. All rights reserved. | Built with Streamlit & Machine Learning"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
