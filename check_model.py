import joblib
import sys

# Load the model
model_path = "models/career_recommender_v1.pkl"
model_pkg = joblib.load(model_path)

print("="*70)
print("Model Information:")
print("="*70)
print(f"Model Name: {model_pkg['model_name']}")
print(f"Version: {model_pkg['version']}")
print(f"Career Names: {model_pkg['career_names']}")
print(f"\nModel Type: {type(model_pkg['model'])}")
print(f"Model: {model_pkg['model']}")

# Try to get sklearn version used to train
import sklearn
print(f"\nCurrent scikit-learn version: {sklearn.__version__}")

# Check numpy version
import numpy as np
print(f"Current numpy version: {np.__version__}")
