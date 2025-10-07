"""
Feature Engineering Module for Career Recommendation Engine
Creates advanced features from user attributes
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


class FeatureEngineer:
    """Advanced feature engineering for career recommendation"""
    
    # Define skill categories
    TECHNICAL_SKILLS = {
        'Python', 'Java', 'SQL', 'Machine Learning', 'Data Analysis',
        'Statistics', 'Cloud Computing', 'UI/UX', 'Excel', 'JavaScript',
        'R', 'Deep Learning', 'NLP', 'Computer Vision', 'DevOps'
    }
    
    SOFT_SKILLS = {
        'Communication', 'Leadership', 'Project Management', 'Business Strategy',
        'Creative Writing', 'Public Speaking', 'Negotiation', 'Team Collaboration',
        'Critical Thinking', 'Problem Solving'
    }
    
    # Define interest categories
    TECH_INTERESTS = {'Technology', 'Science', 'Engineering'}
    CREATIVE_INTERESTS = {'Arts', 'Design', 'Creative', 'Media'}
    BUSINESS_INTERESTS = {'Business', 'Management', 'Finance', 'Entrepreneurship'}
    SOCIAL_INTERESTS = {'Education', 'Health', 'Social', 'Community'}
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer
        
        Args:
            df: Preprocessed DataFrame
        """
        self.df = df.copy()
        
    def parse_skills(self, skills_str: str) -> List[str]:
        """Parse skills string into list"""
        return [s.strip() for s in str(skills_str).split(',') if s.strip()]
    
    def parse_interests(self, interests_str: str) -> List[str]:
        """Parse interests string into list"""
        return [i.strip() for i in str(interests_str).split(',') if i.strip()]
    
    def create_skill_clusters(self) -> pd.DataFrame:
        """
        Create skill cluster features
        
        Returns:
            DataFrame with skill cluster features
        """
        df_features = self.df.copy()
        
        # Parse skills
        df_features['skills_list'] = df_features['skills'].apply(self.parse_skills)
        
        # Count technical skills
        df_features['technical_skills_count'] = df_features['skills_list'].apply(
            lambda skills: sum(1 for s in skills if s in self.TECHNICAL_SKILLS)
        )
        
        # Count soft skills
        df_features['soft_skills_count'] = df_features['skills_list'].apply(
            lambda skills: sum(1 for s in skills if s in self.SOFT_SKILLS)
        )
        
        # Total skills
        df_features['total_skills'] = df_features['skills_list'].apply(len)
        
        # Skill diversity ratio
        df_features['skill_diversity'] = (
            df_features['technical_skills_count'] + df_features['soft_skills_count']
        ) / df_features['total_skills'].replace(0, 1)
        
        # Technical vs Soft skills ratio
        df_features['tech_soft_ratio'] = df_features['technical_skills_count'] / (
            df_features['soft_skills_count'].replace(0, 1)
        )
        
        return df_features
    
    def create_interest_profiles(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create interest profile features
        
        Args:
            df_features: DataFrame with existing features
            
        Returns:
            DataFrame with interest profile features
        """
        # Parse interests
        df_features['interests_list'] = df_features['interests'].apply(self.parse_interests)
        
        # Tech oriented
        df_features['tech_oriented'] = df_features['interests_list'].apply(
            lambda interests: sum(1 for i in interests if i in self.TECH_INTERESTS)
        )
        
        # Creative oriented
        df_features['creative_oriented'] = df_features['interests_list'].apply(
            lambda interests: sum(1 for i in interests if i in self.CREATIVE_INTERESTS)
        )
        
        # Business oriented
        df_features['business_oriented'] = df_features['interests_list'].apply(
            lambda interests: sum(1 for i in interests if i in self.BUSINESS_INTERESTS)
        )
        
        # Social oriented
        df_features['social_oriented'] = df_features['interests_list'].apply(
            lambda interests: sum(1 for i in interests if i in self.SOCIAL_INTERESTS)
        )
        
        # Interest breadth (number of different interest categories)
        df_features['interest_breadth'] = df_features['interests_list'].apply(len)
        
        return df_features
    
    def create_personality_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived personality features
        
        Args:
            df_features: DataFrame with existing features
            
        Returns:
            DataFrame with personality features
        """
        # Already normalized analytical, creative, social from preprocessing
        
        # Dominant personality trait
        personality_cols = ['analytical', 'creative', 'social']
        df_features['dominant_trait'] = df_features[personality_cols].idxmax(axis=1)
        
        # Personality balance (standard deviation of traits)
        df_features['personality_balance'] = df_features[personality_cols].std(axis=1)
        
        # Analytical-Creative balance
        df_features['analytical_creative_balance'] = (
            df_features['analytical'] - df_features['creative']
        )
        
        return df_features
    
    def create_interaction_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different attributes
        
        Args:
            df_features: DataFrame with existing features
            
        Returns:
            DataFrame with interaction features
        """
        # Skills × Education
        df_features['skills_education_score'] = (
            df_features['total_skills'] * df_features['education_encoded']
        )
        
        # Experience × Technical Skills
        df_features['exp_tech_score'] = (
            df_features['experience_normalized'] * df_features['technical_skills_count']
        )
        
        # Analytical × Technical Skills
        df_features['analytical_tech_alignment'] = (
            df_features['analytical'] * df_features['technical_skills_count']
        )
        
        # Creative × Creative Interests
        df_features['creative_alignment'] = (
            df_features['creative'] * df_features['creative_oriented']
        )
        
        # Social × Soft Skills
        df_features['social_skills_alignment'] = (
            df_features['social'] * df_features['soft_skills_count']
        )
        
        return df_features
    
    def create_career_readiness_score(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create a composite career readiness score
        
        Args:
            df_features: DataFrame with existing features
            
        Returns:
            DataFrame with career readiness score
        """
        # Weighted combination of key factors
        df_features['career_readiness'] = (
            0.3 * df_features['total_skills'] / 10 +  # Normalize by max expected skills
            0.25 * df_features['education_encoded'] / 4 +  # Normalize by max education level
            0.25 * df_features['experience_normalized'] +
            0.2 * df_features['personality_balance']
        )
        
        return df_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names for modeling
        
        Returns:
            List of feature column names
        """
        feature_cols = [
            # Basic features
            'analytical', 'creative', 'social',
            'education_encoded', 'experience', 'experience_normalized',
            
            # Skill features
            'technical_skills_count', 'soft_skills_count', 'total_skills',
            'skill_diversity', 'tech_soft_ratio',
            
            # Interest features
            'tech_oriented', 'creative_oriented', 'business_oriented',
            'social_oriented', 'interest_breadth',
            
            # Personality features
            'personality_balance', 'analytical_creative_balance',
            
            # Interaction features
            'skills_education_score', 'exp_tech_score',
            'analytical_tech_alignment', 'creative_alignment',
            'social_skills_alignment',
            
            # Composite features
            'career_readiness'
        ]
        
        return feature_cols
    
    def engineer_all_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Run complete feature engineering pipeline
        
        Returns:
            Tuple of (DataFrame with all features, list of feature names)
        """
        print("Starting feature engineering...")
        
        # Create skill clusters
        df_features = self.create_skill_clusters()
        print("[OK] Skill clusters created")
        
        # Create interest profiles
        df_features = self.create_interest_profiles(df_features)
        print("[OK] Interest profiles created")
        
        # Create personality features
        df_features = self.create_personality_features(df_features)
        print("[OK] Personality features created")
        
        # Create interaction features
        df_features = self.create_interaction_features(df_features)
        print("[OK] Interaction features created")
        
        # Create career readiness score
        df_features = self.create_career_readiness_score(df_features)
        print("[OK] Career readiness score created")
        
        feature_names = self.get_feature_names()
        
        print(f"\nTotal features engineered: {len(feature_names)}")
        print("Feature engineering complete!")
        
        return df_features, feature_names
    
    def get_feature_importance_data(self, df_features: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """
        Prepare data for feature importance analysis
        
        Args:
            df_features: DataFrame with all features
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature statistics
        """
        stats = []
        
        for feature in feature_names:
            stats.append({
                'feature': feature,
                'mean': df_features[feature].mean(),
                'std': df_features[feature].std(),
                'min': df_features[feature].min(),
                'max': df_features[feature].max(),
                'missing': df_features[feature].isnull().sum()
            })
        
        return pd.DataFrame(stats)


if __name__ == "__main__":
    # Example usage
    from preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor("data/synthetic_user_profiles_large.csv")
    df_processed, y_binary, career_names = preprocessor.get_preprocessed_data()
    
    engineer = FeatureEngineer(df_processed)
    df_features, feature_names = engineer.engineer_all_features()
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*50)
    print(f"Total features: {len(feature_names)}")
    print(f"Feature matrix shape: {df_features[feature_names].shape}")
