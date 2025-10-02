# INX Future Inc - Employee Performance Prediction App

A comprehensive Streamlit web application for predicting employee performance using machine learning. This application uses an XGBoost model trained on employee data to predict performance ratings and provide actionable insights.

## 🚀 Features

- **Interactive Prediction Interface**: Easy-to-use form for inputting employee details
- **Real-time Predictions**: Instant performance rating predictions with confidence scores
- **Visual Analytics**: Interactive charts and graphs using Plotly
- **Personalized Recommendations**: Tailored suggestions based on prediction results
- **Model Insights**: Feature importance and department performance analytics
- **Professional UI**: Modern, responsive design with custom CSS styling

## 📋 Prerequisites

- Python 3.8 or higher
- Required Python packages (see requirements_streamlit.txt)
- Trained model files (best_model.pkl, scaler.pkl, all_encoders.pkl)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required packages**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Ensure model files are available**:
   The app looks for model files in these locations (in order):
   - `./data/`
   - `../data/`
   - `D:\Data Science Classes\Machine Learning\Group Assignment\IABAC Exams\Employee Performance Analysis\data`
   
   Required files:
   - `best_model.pkl` - Trained XGBoost model
   - `scaler.pkl` - Feature scaler
   - `all_encoders.pkl` - Label encoders for categorical variables

## 🏃‍♂️ Running the Application

1. **Navigate to the src directory**:
   ```bash
   cd "c:\Users\tonyn\Downloads\IABAC exam elizabeth\src"
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser**:
   The app will automatically open in your default browser, or navigate to `http://localhost:8501`

## 📊 Application Structure

### Pages

1. **🎯 Prediction Page**
   - Employee information input form
   - Real-time performance prediction
   - Confidence scores and probability distribution
   - Personalized recommendations

2. **📈 Model Insights**
   - Feature importance visualization
   - Department performance analytics
   - Key insights and findings

3. **ℹ️ About Page**
   - Project overview
   - Technical details
   - Model performance metrics

### Input Parameters

The prediction model uses these key features:

- **Personal Information**:
  - Age (18-65 years)
  - Total Work Experience (0-40 years)
  - Years in Current Role (0-20 years)
  - Years with Current Manager (0-20 years)
  - Years Since Last Promotion (0-15 years)

- **Work Environment**:
  - Department (Data Science, Development, Finance, HR, R&D, Sales)
  - Job Role (Various technical and managerial roles)
  - Environment Satisfaction (Low, Medium, High, Very High)
  - Work-Life Balance (Bad, Good, Better, Best)
  - Last Salary Hike Percentage (10-25%)

### Performance Ratings

The model predicts three performance levels:
- **Good**: Meets basic expectations
- **Excellent**: Exceeds expectations consistently
- **Outstanding**: Exceptional performance

## 🎨 Customization

### Styling
The app uses custom CSS located in `static/style.css`. You can modify:
- Color schemes and gradients
- Layout and spacing
- Component styling
- Responsive design elements

### Model Configuration
Update the feature mappings and model parameters in `streamlit_app.py`:
- `TOP_10_FEATURES`: List of features used by the model
- `FEATURE_MAPPINGS`: Categorical value mappings
- `PERFORMANCE_MAPPING`: Performance level definitions

## 🔧 Troubleshooting

### Model Files Not Found
If you see "Model files not found" error:
1. Ensure you've run the Jupyter notebook to train and save the model
2. Check that the model files exist in one of the expected directories
3. Update the file paths in `utils.py` if needed

### Package Import Errors
If you encounter import errors:
1. Ensure all packages are installed: `pip install -r requirements_streamlit.txt`
2. Check Python version compatibility
3. Consider using a virtual environment

### Performance Issues
For better performance:
1. Use `@st.cache_resource` for model loading (already implemented)
2. Optimize data processing functions
3. Consider using Streamlit's session state for large datasets

## 📈 Model Performance

The XGBoost model achieves:
- **Accuracy**: 94.2%
- **Precision**: 94.0%
- **Recall**: 94.2%
- **F1-Score**: 94.0%
- **Cross-validation Score**: 93.1%

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the INX Future Inc. employee performance analysis initiative.

## 🆘 Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review the Jupyter notebook for model training details
3. Ensure all dependencies are properly installed

---

**Note**: This application is designed for internal use at INX Future Inc. to assist in employee performance evaluation and HR decision-making.
