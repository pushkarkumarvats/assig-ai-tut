"""
Confidence Scoring Module for Career Recommendation Engine
Implements hybrid confidence scoring with model probabilities and heuristics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter


class ConfidenceScorer:
    """
    Advanced confidence scoring system that combines:
    - Model prediction probabilities
    - Feature-based heuristics
    - Rule-based adjustments for edge cases
    """
    
    def __init__(self, career_names: List[str]):
        """
        Initialize confidence scorer
        
        Args:
            career_names: List of all possible career names
        """
        self.career_names = career_names
        self.career_skill_requirements = self._define_career_requirements()
        
    def _define_career_requirements(self) -> Dict[str, Dict]:
        """
        Define skill and attribute requirements for each career
        
        Returns:
            Dictionary mapping careers to their requirements
        """
        requirements = {
            'Data Scientist': {
                'required_skills': ['Python', 'Statistics', 'Machine Learning', 'Data Analysis'],
                'min_technical_skills': 2,
                'personality_match': {'analytical': 0.6, 'creative': 0.3},
                'min_education': 2  # Bachelor
            },
            'Software Engineer': {
                'required_skills': ['Python', 'Java', 'SQL'],
                'min_technical_skills': 2,
                'personality_match': {'analytical': 0.6},
                'min_education': 2
            },
            'Business Analyst': {
                'required_skills': ['Excel', 'Data Analysis', 'Business Strategy'],
                'min_technical_skills': 1,
                'personality_match': {'analytical': 0.5, 'social': 0.4},
                'min_education': 2
            },
            'Product Manager': {
                'required_skills': ['Project Management', 'Business Strategy', 'Communication'],
                'min_technical_skills': 0,
                'personality_match': {'social': 0.6, 'creative': 0.4},
                'min_education': 2
            },
            'UX Designer': {
                'required_skills': ['UI/UX', 'Creative Writing'],
                'min_technical_skills': 1,
                'personality_match': {'creative': 0.7, 'social': 0.3},
                'min_education': 2
            },
            'Marketing Specialist': {
                'required_skills': ['Communication', 'Creative Writing', 'Business Strategy'],
                'min_technical_skills': 0,
                'personality_match': {'creative': 0.5, 'social': 0.5},
                'min_education': 2
            },
            'Financial Analyst': {
                'required_skills': ['Excel', 'Statistics', 'Business Strategy'],
                'min_technical_skills': 1,
                'personality_match': {'analytical': 0.7},
                'min_education': 2
            },
            'Research Scientist': {
                'required_skills': ['Statistics', 'Python', 'Data Analysis'],
                'min_technical_skills': 2,
                'personality_match': {'analytical': 0.8},
                'min_education': 3  # Master or PhD
            }
        }
        
        return requirements
    
    def calculate_confidence_scores(
        self,
        model_probabilities: np.ndarray,
        user_features: Dict,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Calculate confidence scores for career recommendations
        
        Args:
            model_probabilities: Raw probabilities from model
            user_features: Dictionary of user features
            top_k: Number of top recommendations to return
            
        Returns:
            List of dictionaries with career and confidence scores
        """
        # Step 1: Get model predictions
        career_scores = {}
        
        for i, career in enumerate(self.career_names):
            model_prob = model_probabilities[i]
            
            # Step 2: Calculate feature-based heuristic score
            heuristic_score = self._calculate_heuristic_score(career, user_features)
            
            # Step 3: Combine model probability with heuristic (70% model, 30% heuristic)
            combined_score = 0.7 * model_prob + 0.3 * heuristic_score
            
            # Step 4: Apply rule-based adjustments
            adjusted_score = self._apply_rule_adjustments(
                career, combined_score, user_features
            )
            
            career_scores[career] = adjusted_score
        
        # Step 5: Get top k careers
        sorted_careers = sorted(
            career_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Step 6: Normalize scores to sum to 100%
        total_score = sum([score for _, score in sorted_careers])
        
        recommendations = []
        for career, score in sorted_careers:
            confidence = (score / total_score) * 100 if total_score > 0 else 0
            recommendations.append({
                'title': career,
                'confidence': round(confidence, 2)
            })
        
        return recommendations
    
    def _calculate_heuristic_score(
        self,
        career: str,
        user_features: Dict
    ) -> float:
        """
        Calculate heuristic-based score based on career requirements
        
        Args:
            career: Career name
            user_features: User feature dictionary
            
        Returns:
            Heuristic score between 0 and 1
        """
        if career not in self.career_skill_requirements:
            return 0.5  # Default score for unknown careers
        
        requirements = self.career_skill_requirements[career]
        score = 0.0
        weights = []
        
        # Check skill match (weight: 0.4)
        user_skills = set(user_features.get('skills', []))
        required_skills = set(requirements.get('required_skills', []))
        
        if required_skills:
            skill_match = len(user_skills & required_skills) / len(required_skills)
            score += 0.4 * skill_match
            weights.append(0.4)
        
        # Check technical skills requirement (weight: 0.2)
        tech_skills_count = user_features.get('technical_skills_count', 0)
        min_tech_skills = requirements.get('min_technical_skills', 0)
        
        if tech_skills_count >= min_tech_skills:
            score += 0.2
        else:
            score += 0.2 * (tech_skills_count / max(min_tech_skills, 1))
        weights.append(0.2)
        
        # Check personality match (weight: 0.25)
        personality_match = requirements.get('personality_match', {})
        personality_score = 0.0
        
        for trait, required_value in personality_match.items():
            user_value = user_features.get(trait, 0.5)
            # Score based on how close user trait is to requirement
            trait_score = 1 - abs(user_value - required_value)
            personality_score += trait_score
        
        if personality_match:
            personality_score /= len(personality_match)
            score += 0.25 * personality_score
            weights.append(0.25)
        
        # Check education requirement (weight: 0.15)
        user_education = user_features.get('education_encoded', 2)
        min_education = requirements.get('min_education', 2)
        
        if user_education >= min_education:
            score += 0.15
        else:
            score += 0.15 * (user_education / min_education)
        weights.append(0.15)
        
        # Normalize by total weight used
        total_weight = sum(weights)
        return score / total_weight if total_weight > 0 else 0.5
    
    def _apply_rule_adjustments(
        self,
        career: str,
        score: float,
        user_features: Dict
    ) -> float:
        """
        Apply rule-based adjustments for edge cases
        
        Args:
            career: Career name
            score: Current confidence score
            user_features: User features
            
        Returns:
            Adjusted confidence score
        """
        adjusted_score = score
        
        # Rule 1: Boost for high experience in technical careers
        if career in ['Data Scientist', 'Software Engineer', 'Research Scientist']:
            experience = user_features.get('experience', 0)
            if experience >= 5:
                adjusted_score *= 1.1  # 10% boost
        
        # Rule 2: Boost for creative careers with high creative personality
        if career in ['UX Designer', 'Marketing Specialist']:
            creative_score = user_features.get('creative', 0.5)
            if creative_score > 0.7:
                adjusted_score *= 1.15  # 15% boost
        
        # Rule 3: Penalize technical careers without technical skills
        if career in ['Data Scientist', 'Software Engineer']:
            tech_skills = user_features.get('technical_skills_count', 0)
            if tech_skills == 0:
                adjusted_score *= 0.5  # 50% penalty
        
        # Rule 4: Boost for management careers with leadership skills
        if career in ['Product Manager', 'Business Analyst']:
            user_skills = user_features.get('skills', [])
            if 'Leadership' in user_skills or 'Project Management' in user_skills:
                adjusted_score *= 1.2  # 20% boost
        
        # Rule 5: Boost for alignment of interests
        if career == 'Data Scientist' and user_features.get('tech_oriented', 0) > 2:
            adjusted_score *= 1.1
        
        if career == 'Marketing Specialist' and user_features.get('business_oriented', 0) > 2:
            adjusted_score *= 1.1
        
        # Ensure score stays between 0 and 1
        return min(max(adjusted_score, 0.0), 1.0)
    
    def validate_confidence_scores(
        self,
        predictions: List[Dict],
        actual_careers: List[str]
    ) -> Dict[str, float]:
        """
        Validate confidence scores against known career paths
        
        Args:
            predictions: List of predicted careers with confidence scores
            actual_careers: List of actual career names
            
        Returns:
            Dictionary of validation metrics
        """
        # Check if top prediction is in actual careers
        top_prediction = predictions[0]['title'] if predictions else None
        top_match = 1.0 if top_prediction in actual_careers else 0.0
        
        # Check if any of top 3 predictions are in actual careers
        top_3_predictions = [p['title'] for p in predictions[:3]]
        top_3_match = any(pred in actual_careers for pred in top_3_predictions)
        
        # Calculate average confidence for correct predictions
        correct_confidences = [
            p['confidence'] for p in predictions
            if p['title'] in actual_careers
        ]
        avg_correct_confidence = np.mean(correct_confidences) if correct_confidences else 0.0
        
        # Calculate average confidence for incorrect predictions
        incorrect_confidences = [
            p['confidence'] for p in predictions
            if p['title'] not in actual_careers
        ]
        avg_incorrect_confidence = np.mean(incorrect_confidences) if incorrect_confidences else 0.0
        
        return {
            'top_1_accuracy': top_match,
            'top_3_accuracy': float(top_3_match),
            'avg_confidence_correct': avg_correct_confidence,
            'avg_confidence_incorrect': avg_incorrect_confidence,
            'confidence_separation': avg_correct_confidence - avg_incorrect_confidence
        }
    
    def generate_confidence_report(
        self,
        predictions: List[Dict],
        user_features: Dict
    ) -> str:
        """
        Generate detailed confidence report
        
        Args:
            predictions: List of career predictions
            user_features: User features
            
        Returns:
            Formatted confidence report string
        """
        report = "\n" + "="*70 + "\n"
        report += "CONFIDENCE SCORE REPORT\n"
        report += "="*70 + "\n\n"
        
        report += "User Profile Summary:\n"
        report += f"  Skills: {', '.join(user_features.get('skills', []))}\n"
        report += f"  Education: {user_features.get('education', 'Unknown')}\n"
        report += f"  Experience: {user_features.get('experience', 0)} years\n"
        report += f"  Analytical: {user_features.get('analytical', 0.5):.2f}\n"
        report += f"  Creative: {user_features.get('creative', 0.5):.2f}\n"
        report += f"  Social: {user_features.get('social', 0.5):.2f}\n\n"
        
        report += "Top Career Recommendations:\n"
        report += "-" * 70 + "\n"
        
        for i, pred in enumerate(predictions, 1):
            report += f"{i}. {pred['title']}\n"
            report += f"   Confidence: {pred['confidence']:.2f}%\n"
            
            # Add reasoning
            if pred['title'] in self.career_skill_requirements:
                requirements = self.career_skill_requirements[pred['title']]
                report += f"   Key Requirements: {', '.join(requirements['required_skills'][:3])}\n"
            
            report += "\n"
        
        report += "="*70 + "\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    career_names = [
        'Data Scientist', 'Software Engineer', 'Business Analyst',
        'Product Manager', 'UX Designer', 'Marketing Specialist',
        'Financial Analyst', 'Research Scientist'
    ]
    
    scorer = ConfidenceScorer(career_names)
    
    # Test with sample user
    model_probs = np.array([0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])
    
    user_features = {
        'skills': ['Python', 'Machine Learning', 'Statistics'],
        'technical_skills_count': 3,
        'analytical': 0.8,
        'creative': 0.4,
        'social': 0.6,
        'education_encoded': 3,
        'education': 'Master',
        'experience': 4,
        'tech_oriented': 3
    }
    
    recommendations = scorer.calculate_confidence_scores(
        model_probs, user_features, top_k=5
    )
    
    print(scorer.generate_confidence_report(recommendations, user_features))
