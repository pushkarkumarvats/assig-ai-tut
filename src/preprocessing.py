"""
Data Preprocessing Module for Career Recommendation Engine
Handles data loading, cleaning, and initial transformations
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


class DataPreprocessor:
    """Handles data loading and preprocessing operations"""
    
    def __init__(self, data_path: str):
        """
        Initialize preprocessor with data path
        
        Args:
            data_path: Path to CSV file containing user profiles
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data inspection"""
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nData types:\n{self.df.dtypes}")
        return self.df
    
    def parse_list_column(self, column: str) -> List[List[str]]:
        """
        Parse comma-separated string columns into lists
        
        Args:
            column: Column name to parse
            
        Returns:
            List of lists containing parsed values
        """
        return self.df[column].apply(
            lambda x: [item.strip() for item in str(x).split(',') if item.strip()]
        ).tolist()
    
    def get_unique_values(self, column: str) -> set:
        """
        Get unique values from list-type columns
        
        Args:
            column: Column name
            
        Returns:
            Set of unique values
        """
        parsed = self.parse_list_column(column)
        return set([item for sublist in parsed for item in sublist])
    
    def normalize_personality_traits(self) -> pd.DataFrame:
        """
        Normalize personality trait scores (analytical, creative, social)
        
        Returns:
            DataFrame with normalized personality traits
        """
        personality_cols = ['analytical', 'creative', 'social']
        df_normalized = self.df.copy()
        
        # Ensure values are in [0, 1] range
        for col in personality_cols:
            df_normalized[col] = df_normalized[col].clip(0, 1)
        
        # Standardize for ML models
        df_normalized[personality_cols] = self.scaler.fit_transform(
            df_normalized[personality_cols]
        )
        
        return df_normalized
    
    def encode_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode education level with ordinal mapping
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded education
        """
        education_mapping = {
            'High School': 1,
            'Bachelor': 2,
            'Master': 3,
            'PhD': 4
        }
        
        df['education_encoded'] = df['education'].map(education_mapping)
        return df
    
    def create_experience_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create experience level bins for better categorization
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with experience bins
        """
        bins = [-1, 0, 2, 5, 10, 100]
        labels = ['Entry', 'Junior', 'Mid', 'Senior', 'Expert']
        df['experience_level'] = pd.cut(df['experience'], bins=bins, labels=labels)
        
        # Also keep normalized experience
        df['experience_normalized'] = df['experience'] / df['experience'].max()
        
        return df
    
    def prepare_target_labels(self) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare multi-label target variable
        
        Returns:
            Tuple of (binary label matrix, career names)
        """
        target_careers = self.parse_list_column('target_careers')
        y_binary = self.mlb.fit_transform(target_careers)
        career_names = self.mlb.classes_
        
        print(f"\nTotal unique careers: {len(career_names)}")
        print(f"Career distribution:")
        career_counts = pd.Series(
            [item for sublist in target_careers for item in sublist]
        ).value_counts()
        print(career_counts)
        
        return y_binary, list(career_names)
    
    def analyze_class_imbalance(self, y_binary: np.ndarray, career_names: List[str]) -> pd.DataFrame:
        """
        Analyze class imbalance in target labels
        
        Args:
            y_binary: Binary label matrix
            career_names: List of career names
            
        Returns:
            DataFrame with class distribution statistics
        """
        class_distribution = pd.DataFrame({
            'career': career_names,
            'count': y_binary.sum(axis=0),
            'percentage': (y_binary.sum(axis=0) / len(y_binary)) * 100
        }).sort_values('count', ascending=False)
        
        print("\nClass Distribution Analysis:")
        print(class_distribution)
        print(f"\nImbalance Ratio (max/min): {class_distribution['count'].max() / class_distribution['count'].min():.2f}")
        
        return class_distribution
    
    def get_preprocessed_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Complete preprocessing pipeline
        
        Returns:
            Tuple of (processed DataFrame, target labels, career names)
        """
        if self.df is None:
            self.load_data()
        
        # Normalize personality traits
        df_processed = self.normalize_personality_traits()
        
        # Encode categorical variables
        df_processed = self.encode_education(df_processed)
        df_processed = self.create_experience_bins(df_processed)
        
        # Prepare target labels
        y_binary, career_names = self.prepare_target_labels()
        
        # Analyze class imbalance
        self.analyze_class_imbalance(y_binary, career_names)
        
        return df_processed, y_binary, career_names


def split_data(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets
    
    Args:
        X: Feature DataFrame
        y: Target labels
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor("data/synthetic_user_profiles_large.csv")
    df_processed, y_binary, career_names = preprocessor.get_preprocessed_data()
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Target matrix shape: {y_binary.shape}")
    print(f"Number of careers: {len(career_names)}")
