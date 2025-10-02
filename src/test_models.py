"""
Test script to verify model loading
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

try:
    import joblib
    import pandas as pd
    import numpy as np
    
    print("✅ Basic imports successful")
    
    # Test model loading
    data_path = Path("../data")
    print(f"📁 Data directory exists: {data_path.exists()}")
    
    if data_path.exists():
        model_files = {
            'model': data_path / 'best_model.pkl',
            'scaler': data_path / 'scaler.pkl',
            'encoders': data_path / 'all_encoders.pkl'
        }
        
        for name, file_path in model_files.items():
            exists = file_path.exists()
            print(f"📄 {name}: {file_path.name} - {'✅ Found' if exists else '❌ Missing'}")
            
            if exists:
                try:
                    obj = joblib.load(file_path)
                    print(f"   🔍 Type: {type(obj).__name__}")
                    if hasattr(obj, 'feature_importances_'):
                        print(f"   📊 Features: {len(obj.feature_importances_)}")
                    elif hasattr(obj, 'classes_'):
                        print(f"   🏷️  Classes: {obj.classes_}")
                except Exception as e:
                    print(f"   ❌ Load error: {e}")
        
        print("\n🎯 Model loading test completed!")
        
    else:
        print("❌ Data directory not found")
        print("💡 Make sure you're running this from the src directory")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Install required packages: pip install -r requirements_streamlit.txt")
