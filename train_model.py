"""
Complete model training script
Runs all preprocessing, feature engineering, and model training
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*70)
print("CAREER RECOMMENDATION ENGINE - MODEL TRAINING")
print("="*70)
print("\nThis will take approximately 10-15 minutes...")
print("\nStep 1: Data Preprocessing & Feature Engineering")
print("-"*70)

# Import modules
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_trainer import CareerModelTrainer

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess data
print("\n[LOADING] Dataset...")
preprocessor = DataPreprocessor('data/synthetic_user_profiles_large.csv')
df_processed, y_binary, career_names = preprocessor.get_preprocessed_data()
print(f"[OK] Data loaded: {df_processed.shape[0]} samples")

# Step 2: Feature engineering
print("\n[ENGINEERING] Features...")
engineer = FeatureEngineer(df_processed)
df_features, feature_names = engineer.engineer_all_features()
print(f"[OK] Created {len(feature_names)} features")

# Step 3: Prepare feature matrix
print("\n[PREPARING] Feature matrix...")
X = df_features[feature_names]
print(f"[OK] Feature matrix shape: {X.shape}")

# Step 4: Train-test split
print("\n[SPLITTING] Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)
print(f"[OK] Training set: {X_train.shape}")
print(f"[OK] Test set: {X_test.shape}")

# Step 5: Create output directories
print("\n[CREATING] Output directories...")
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
print("[OK] Directories created")

# Step 6: Save processed data
print("\n[SAVING] Processed data...")
X.to_csv('data/processed/features.csv', index=False)
np.save('data/processed/targets.npy', y_binary)

metadata = {
    'feature_names': feature_names,
    'career_names': career_names,
    'n_samples': len(X),
    'n_features': len(feature_names),
    'n_careers': len(career_names)
}
joblib.dump(metadata, 'data/processed/metadata.pkl')
print("[OK] Processed data saved")

# Step 7: Train models
print("\n" + "="*70)
print("Step 2: Model Training")
print("="*70)

trainer = CareerModelTrainer(random_state=42)

# Train Random Forest
print("\n[TRAINING] Random Forest (this may take 5-10 minutes)...")
rf_model, rf_metrics = trainer.train_random_forest(
    X_train.values, y_train, optimize=True
)

# Train XGBoost
print("\n[TRAINING] XGBoost (this may take 5-10 minutes)...")
xgb_model, xgb_metrics = trainer.train_xgboost(
    X_train.values, y_train, optimize=True
)

# Step 8: Compare models
print("\n[COMPARING] Models on test set...")
comparison_df = trainer.compare_models(X_test.values, y_test)
print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
print(comparison_df.to_string(index=False))

# Step 9: Save best model
print("\n[SAVING] Best model...")
trainer.career_names = career_names  # Set career names before saving
trainer.save_model('models/career_recommender_v1.pkl')

# Step 10: Save artifacts
artifacts = {
    'feature_names': feature_names,
    'career_names': career_names,
    'test_metrics': comparison_df.to_dict(),
    'model_version': '1.0'
}
joblib.dump(artifacts, 'models/model_artifacts_v1.pkl')
print("[OK] Model artifacts saved")

# Final summary
print("\n" + "="*70)
print("[SUCCESS] TRAINING COMPLETE!")
print("="*70)
print(f"\n[BEST] Model: {trainer.best_model_name}")
print(f"\n[FILE] Model saved to: models/career_recommender_v1.pkl")
print(f"[METRICS] Test Performance:")
best_row = comparison_df[comparison_df['model'] == trainer.best_model_name].iloc[0]
print(f"   - Hamming Loss: {best_row['hamming_loss']:.4f}")
print(f"   - Label Ranking Avg Precision: {best_row['label_ranking_avg_precision']:.4f}")
print(f"   - Precision@3: {best_row['precision_at_3']:.4f}")
print(f"   - Subset Accuracy: {best_row['subset_accuracy']:.4f}")

print("\n[READY] You can now deploy to Render!")
print("="*70)
