"""
Script to test the Career Recommendation API with sample profiles
"""

import requests
import json
from typing import Dict, List

API_URL = "https://career-recommender-api-gpbq.onrender.com"


def print_recommendations(recommendations: List[Dict], profile_name: str):
    """Print recommendations in a formatted way"""
    print(f"\n{'='*70}")
    print(f"Career Recommendations for: {profile_name}")
    print(f"{'='*70}\n")
    
    for i, career in enumerate(recommendations, 1):
        bar_length = int(career['confidence'] / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        print(f"{i}. {career['title']:<25} {career['confidence']:>6.2f}% {bar}")
    
    print(f"\n{'='*70}\n")


def test_profile(profile: Dict, profile_name: str):
    """Test a single profile"""
    try:
        response = requests.post(f"{API_URL}/predict", json=profile)
        
        if response.status_code == 200:
            data = response.json()
            print_recommendations(data['careers'], profile_name)
            print(f"Model Version: {data['model_version']}")
            print(f"Timestamp: {data['timestamp']}")
        else:
            print(f"\nError {response.status_code}: {response.json()}")
    
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API. Make sure the server is running.")
        print("   Run: python run_api.py")
    except Exception as e:
        print(f"\nError: {e}")


def main():
    """Test multiple sample profiles"""
    # Check if API is running
    try:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"\n[OK] API is running")
            print(f"[OK] Status: {health_data['status']}")
            print(f"[OK] Model loaded: {health_data['model_loaded']}")
        else:
            print("\n[WARNING] API is running but health check failed")
    except:
        print("\n[ERROR] API is not running. Please start it first with: python run_api.py")
        return
    # Test Profile 1: Data Scientist Profile
    profile_1 = {
        "skills": ["Python", "Machine Learning", "Statistics", "SQL", "Data Analysis"],
        "interests": ["Technology", "Science"],
        "personality": {
            "analytical": 0.9,
            "creative": 0.3,
            "social": 0.4
        },
        "education": "Master",
        "experience": 5
    }
    test_profile(profile_1, "Data Scientist Profile (Highly Technical)")
    
    # Test Profile 2: Creative Designer Profile
    profile_2 = {
        "skills": ["UI/UX", "Creative Writing", "Communication"],
        "interests": ["Arts", "Design", "Technology"],
        "personality": {
            "analytical": 0.3,
            "creative": 0.9,
            "social": 0.7
        },
        "education": "Bachelor",
        "experience": 3
    }
    test_profile(profile_2, "UX Designer Profile (Highly Creative)")
    
    # Test Profile 3: Business Manager Profile
    profile_3 = {
        "skills": ["Leadership", "Project Management", "Business Strategy", "Communication"],
        "interests": ["Business", "Management"],
        "personality": {
            "analytical": 0.6,
            "creative": 0.5,
            "social": 0.9
        },
        "education": "Master",
        "experience": 8
    }
    test_profile(profile_3, "Product Manager Profile (Highly Social)")
    
    # Test Profile 4: Entry Level Graduate
    profile_4 = {
        "skills": ["Python", "Communication"],
        "interests": ["Technology", "Education"],
        "personality": {
            "analytical": 0.6,
            "creative": 0.5,
            "social": 0.6
        },
        "education": "Bachelor",
        "experience": 0
    }
    test_profile(profile_4, "Entry Level Graduate (Balanced)")
    
    # Test Profile 5: Finance Analyst
    profile_5 = {
        "skills": ["Excel", "Statistics", "Data Analysis", "Business Strategy"],
        "interests": ["Business", "Finance"],
        "personality": {
            "analytical": 0.9,
            "creative": 0.2,
            "social": 0.5
        },
        "education": "Bachelor",
        "experience": 4
    }
    test_profile(profile_5, "Financial Analyst Profile (Analytical)")
    
    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
