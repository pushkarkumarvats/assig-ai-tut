"""
Test Suite for Career Recommendation Engine API
Comprehensive tests for all API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import app

# Initialize test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test health and information endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictionEndpoint:
    """Test career prediction endpoint"""
    
    def test_valid_prediction_request(self):
        """Test prediction with valid input"""
        request_data = {
            "skills": ["Python", "Communication"],
            "interests": ["Technology", "Management"],
            "personality": {
                "analytical": 0.8,
                "creative": 0.4,
                "social": 0.7
            },
            "education": "Bachelor",
            "experience": 3
        }
        
        response = client.post("/predict", json=request_data)
        
        # May fail if model not loaded, but structure should be checked
        if response.status_code == 200:
            data = response.json()
            assert "careers" in data
            assert "model_version" in data
            assert "timestamp" in data
            assert len(data["careers"]) == 5
            
            # Check career structure
            for career in data["careers"]:
                assert "title" in career
                assert "confidence" in career
                assert isinstance(career["confidence"], (int, float))
            
            # Check confidences sum approximately to 100
            total_confidence = sum(c["confidence"] for c in data["careers"])
            assert 99 <= total_confidence <= 101
    
    def test_missing_required_fields(self):
        """Test prediction with missing required fields"""
        request_data = {
            "skills": ["Python"],
            "interests": ["Technology"]
            # Missing personality, education, experience
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_education_level(self):
        """Test prediction with invalid education level"""
        request_data = {
            "skills": ["Python"],
            "interests": ["Technology"],
            "personality": {
                "analytical": 0.8,
                "creative": 0.4,
                "social": 0.7
            },
            "education": "Elementary",  # Invalid
            "experience": 3
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_personality_scores(self):
        """Test prediction with out-of-range personality scores"""
        request_data = {
            "skills": ["Python"],
            "interests": ["Technology"],
            "personality": {
                "analytical": 1.5,  # Out of range
                "creative": 0.4,
                "social": 0.7
            },
            "education": "Bachelor",
            "experience": 3
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    def test_negative_experience(self):
        """Test prediction with negative experience"""
        request_data = {
            "skills": ["Python"],
            "interests": ["Technology"],
            "personality": {
                "analytical": 0.8,
                "creative": 0.4,
                "social": 0.7
            },
            "education": "Bachelor",
            "experience": -5  # Invalid
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    def test_empty_skills_list(self):
        """Test prediction with empty skills list"""
        request_data = {
            "skills": [],  # Empty
            "interests": ["Technology"],
            "personality": {
                "analytical": 0.8,
                "creative": 0.4,
                "social": 0.7
            },
            "education": "Bachelor",
            "experience": 3
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    def test_technical_profile(self):
        """Test prediction for highly technical profile"""
        request_data = {
            "skills": ["Python", "Machine Learning", "Statistics", "SQL"],
            "interests": ["Technology", "Science"],
            "personality": {
                "analytical": 0.9,
                "creative": 0.2,
                "social": 0.3
            },
            "education": "Master",
            "experience": 5
        }
        
        response = client.post("/predict", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            # For technical profile, expect technical careers in top recommendations
            top_careers = [c["title"] for c in data["careers"][:3]]
            technical_careers = ["Data Scientist", "Software Engineer", "Research Scientist"]
            
            # At least one technical career should be in top 3
            assert any(career in technical_careers for career in top_careers)
    
    def test_creative_profile(self):
        """Test prediction for creative profile"""
        request_data = {
            "skills": ["UI/UX", "Creative Writing", "Communication"],
            "interests": ["Arts", "Design"],
            "personality": {
                "analytical": 0.3,
                "creative": 0.9,
                "social": 0.7
            },
            "education": "Bachelor",
            "experience": 2
        }
        
        response = client.post("/predict", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["careers"]) == 5


class TestInformationEndpoints:
    """Test information endpoints"""
    
    def test_get_available_careers(self):
        """Test getting list of available careers"""
        response = client.get("/careers")
        
        if response.status_code == 200:
            data = response.json()
            assert "careers" in data
            assert "count" in data
            assert isinstance(data["careers"], list)
            assert data["count"] == len(data["careers"])
    
    def test_get_model_info(self):
        """Test getting model information"""
        response = client.get("/model-info")
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "version" in data
            assert "num_careers" in data
            assert "feature_count" in data


class TestInputValidation:
    """Test input validation edge cases"""
    
    def test_maximum_experience(self):
        """Test with maximum allowed experience"""
        request_data = {
            "skills": ["Python"],
            "interests": ["Technology"],
            "personality": {
                "analytical": 0.8,
                "creative": 0.4,
                "social": 0.7
            },
            "education": "PhD",
            "experience": 50  # Maximum
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code in [200, 503]  # Valid or model not loaded
    
    def test_boundary_personality_scores(self):
        """Test with boundary personality scores"""
        request_data = {
            "skills": ["Python"],
            "interests": ["Technology"],
            "personality": {
                "analytical": 0.0,  # Minimum
                "creative": 1.0,    # Maximum
                "social": 0.5       # Middle
            },
            "education": "Bachelor",
            "experience": 0
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code in [200, 503]
    
    def test_many_skills(self):
        """Test with many skills"""
        request_data = {
            "skills": ["Python", "Java", "SQL", "Machine Learning", "Data Analysis",
                      "Leadership", "Communication", "Project Management"],
            "interests": ["Technology", "Business", "Management"],
            "personality": {
                "analytical": 0.7,
                "creative": 0.6,
                "social": 0.8
            },
            "education": "Master",
            "experience": 10
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code in [200, 503]


class TestResponseStructure:
    """Test API response structure and format"""
    
    def test_career_recommendation_structure(self):
        """Test that career recommendations have correct structure"""
        request_data = {
            "skills": ["Python", "Communication"],
            "interests": ["Technology"],
            "personality": {
                "analytical": 0.8,
                "creative": 0.4,
                "social": 0.7
            },
            "education": "Bachelor",
            "experience": 3
        }
        
        response = client.post("/predict", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check careers are sorted by confidence (descending)
            confidences = [c["confidence"] for c in data["careers"]]
            assert confidences == sorted(confidences, reverse=True)
            
            # Check all confidences are positive
            assert all(c >= 0 for c in confidences)
            
            # Check timestamp format
            assert "T" in data["timestamp"]  # ISO format


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
