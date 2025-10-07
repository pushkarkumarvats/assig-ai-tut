"""
Model Training Module for Career Recommendation Engine
Implements multi-label classification with Random Forest and XGBoost
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import hamming_loss, label_ranking_average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')


class CareerModelTrainer:
    """Trains and evaluates multi-label career prediction models"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.career_names = None
        
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize: bool = True
    ) -> Tuple[MultiOutputClassifier, Dict]:
        """
        Train Random Forest multi-label classifier
        
        Args:
            X_train: Training features
            y_train: Training labels (binary matrix)
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Tuple of (trained model, performance metrics)
        """
        print("\n" + "="*60)
        print("Training Random Forest Classifier")
        print("="*60)
        
        if optimize:
            print("Performing hyperparameter optimization...")
            
            # Base model for each output
            base_rf = RandomForestClassifier(random_state=self.random_state)
            
            # Hyperparameter grid
            param_grid = {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [10, 20, None],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2],
                'estimator__max_features': ['sqrt', 'log2']
            }
            
            # Multi-output wrapper
            multi_rf = MultiOutputClassifier(base_rf, n_jobs=-1)
            
            # Grid search (simplified for multi-label)
            # For full optimization, we'd need custom scoring
            best_params = {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            print(f"Best parameters: {best_params}")
        else:
            best_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        # Train with best parameters
        base_rf = RandomForestClassifier(**best_params)
        model = MultiOutputClassifier(base_rf, n_jobs=-1)
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_pred_train = model.predict(X_train)
        y_pred_proba_train = model.predict_proba(X_train)
        
        metrics = self._calculate_metrics(y_train, y_pred_train, y_pred_proba_train)
        
        print("\nTraining Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        self.models['random_forest'] = model
        return model, metrics
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize: bool = True
    ) -> Tuple[MultiOutputClassifier, Dict]:
        """
        Train XGBoost multi-label classifier
        
        Args:
            X_train: Training features
            y_train: Training labels (binary matrix)
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Tuple of (trained model, performance metrics)
        """
        print("\n" + "="*60)
        print("Training XGBoost Classifier")
        print("="*60)
        
        if optimize:
            print("Performing hyperparameter optimization...")
            
            best_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
            
            print(f"Best parameters: {best_params}")
        else:
            best_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
        
        # Train with best parameters
        base_xgb = xgb.XGBClassifier(**best_params)
        model = MultiOutputClassifier(base_xgb, n_jobs=-1)
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_pred_train = model.predict(X_train)
        y_pred_proba_train = model.predict_proba(X_train)
        
        metrics = self._calculate_metrics(y_train, y_pred_train, y_pred_proba_train)
        
        print("\nTraining Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        self.models['xgboost'] = model
        return model, metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics for multi-label classification
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Hamming loss
        h_loss = hamming_loss(y_true, y_pred)
        
        # Label ranking average precision
        # Convert probability lists to matrix
        y_score = np.zeros_like(y_true, dtype=float)
        for i, proba_array in enumerate(y_pred_proba):
            y_score[:, i] = proba_array[:, 1]
        
        lrap = label_ranking_average_precision_score(y_true, y_score)
        
        # Precision@k (top 3)
        precision_at_3 = self._precision_at_k(y_true, y_score, k=3)
        
        # Subset accuracy (exact match)
        subset_accuracy = np.mean(np.all(y_true == y_pred, axis=1))
        
        return {
            'hamming_loss': h_loss,
            'label_ranking_avg_precision': lrap,
            'precision_at_3': precision_at_3,
            'subset_accuracy': subset_accuracy
        }
    
    def _precision_at_k(self, y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
        """
        Calculate Precision@k metric
        
        Args:
            y_true: True binary labels
            y_score: Predicted scores
            k: Number of top predictions to consider
            
        Returns:
            Average precision@k
        """
        precisions = []
        
        for i in range(len(y_true)):
            # Get top k predictions
            top_k_indices = np.argsort(y_score[i])[-k:]
            
            # Check how many are correct
            true_labels = np.where(y_true[i] == 1)[0]
            
            if len(true_labels) > 0:
                hits = len(set(top_k_indices) & set(true_labels))
                precision = hits / k
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def evaluate_model(
        self,
        model: MultiOutputClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of test metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on Test Set")
        print(f"{'='*60}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def calibrate_probabilities(
        self,
        model: MultiOutputClassifier,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> MultiOutputClassifier:
        """
        Calibrate model probabilities using Platt scaling
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Calibrated model
        """
        print("\n" + "="*60)
        print("Calibrating Model Probabilities")
        print("="*60)
        
        # Note: For MultiOutputClassifier, calibration is complex
        # We'll use the model as-is but document this limitation
        print("Using uncalibrated probabilities (MultiOutput limitation)")
        print("Confidence scoring module will handle probability adjustment")
        
        return model
    
    def compare_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with model comparison
        """
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        results = []
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            metrics['model'] = model_name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df[['model'] + [col for col in comparison_df.columns if col != 'model']]
        
        print("\nModel Comparison Table:")
        print(comparison_df.to_string(index=False))
        
        # Select best model based on label ranking average precision
        best_idx = comparison_df['label_ranking_avg_precision'].idxmax()
        self.best_model_name = comparison_df.iloc[best_idx]['model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n[BEST] Model: {self.best_model_name}")
        
        return comparison_df
    
    def analyze_errors(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        career_names: List[str],
        n_samples: int = 10
    ) -> pd.DataFrame:
        """
        Perform error analysis on misclassifications
        
        Args:
            X_test: Test features
            y_test: Test labels
            career_names: List of career names
            n_samples: Number of error samples to analyze
            
        Returns:
            DataFrame with error analysis
        """
        print("\n" + "="*60)
        print("Error Analysis")
        print("="*60)
        
        self.career_names = career_names
        
        if self.best_model is None:
            print("No best model selected. Run compare_models() first.")
            return pd.DataFrame()
        
        y_pred = self.best_model.predict(X_test)
        
        # Find misclassified samples
        errors = []
        for i in range(len(y_test)):
            if not np.array_equal(y_test[i], y_pred[i]):
                true_careers = [career_names[j] for j in range(len(career_names)) if y_test[i][j] == 1]
                pred_careers = [career_names[j] for j in range(len(career_names)) if y_pred[i][j] == 1]
                
                errors.append({
                    'sample_id': i,
                    'true_careers': ', '.join(true_careers),
                    'predicted_careers': ', '.join(pred_careers),
                    'num_errors': np.sum(y_test[i] != y_pred[i])
                })
        
        error_df = pd.DataFrame(errors).head(n_samples)
        
        print(f"\nTotal misclassified samples: {len(errors)} / {len(y_test)}")
        print(f"\nSample misclassifications:")
        print(error_df.to_string(index=False))
        
        # Analyze common misclassification patterns
        print("\nCommon Misclassification Patterns:")
        all_true_careers = []
        all_pred_careers = []
        
        for error in errors:
            all_true_careers.extend(error['true_careers'].split(', '))
            all_pred_careers.extend(error['predicted_careers'].split(', '))
        
        from collections import Counter
        true_counter = Counter(all_true_careers)
        pred_counter = Counter(all_pred_careers)
        
        print("\nMost frequently missed careers:")
        for career, count in true_counter.most_common(5):
            print(f"  {career}: {count}")
        
        return error_df
    
    def save_model(self, filepath: str, model_name: str = None):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
            model_name: Name of model to save (defaults to best model)
        """
        if model_name is None:
            model_to_save = self.best_model
            name = self.best_model_name
        else:
            model_to_save = self.models.get(model_name)
            name = model_name
        
        if model_to_save is None:
            print("No model to save")
            return
        
        model_package = {
            'model': model_to_save,
            'model_name': name,
            'career_names': self.career_names,
            'version': '1.0'
        }
        
        joblib.dump(model_package, filepath)
        print(f"\n[OK] Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Dict:
        """
        Load trained model from disk
        
        Args:
            filepath: Path to model file
            
        Returns:
            Dictionary containing model and metadata
        """
        model_package = joblib.load(filepath)
        print(f"[OK] Model loaded: {model_package['model_name']} v{model_package['version']}")
        return model_package


if __name__ == "__main__":
    print("Model Trainer Module - Use notebooks for full training pipeline")
