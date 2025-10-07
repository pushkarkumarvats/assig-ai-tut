"""
FastAPI Deployment for Career Recommendation Engine
Provides REST API endpoint for career predictions with confidence scores
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import numpy as np
import joblib
import os
from datetime import datetime
import logging

try:
    from .confidence_scorer import ConfidenceScorer
except ImportError:
    from confidence_scorer import ConfidenceScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Career Recommendation Engine API",
    description="AI-powered career recommendation system with confidence scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scorer
model_package = None
confidence_scorer = None
feature_names = None

# ==================== Request/Response Models ====================

class PersonalityScores(BaseModel):
    """Personality trait scores"""
    analytical: float = Field(..., ge=0.0, le=1.0, description="Analytical trait score (0-1)")
    creative: float = Field(..., ge=0.0, le=1.0, description="Creative trait score (0-1)")
    social: float = Field(..., ge=0.0, le=1.0, description="Social trait score (0-1)")


class PredictionRequest(BaseModel):
    """Request model for career prediction"""
    skills: List[str] = Field(..., min_items=1, description="List of user skills")
    interests: List[str] = Field(..., min_items=1, description="List of user interests")
    personality: PersonalityScores = Field(..., description="Personality trait scores")
    education: str = Field(..., description="Education level: High School, Bachelor, Master, or PhD")
    experience: int = Field(..., ge=0, le=50, description="Years of work experience")
    
    @validator('education')
    def validate_education(cls, v):
        valid_levels = ['High School', 'Bachelor', 'Master', 'PhD']
        if v not in valid_levels:
            raise ValueError(f"Education must be one of: {', '.join(valid_levels)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class CareerRecommendation(BaseModel):
    """Single career recommendation"""
    title: str = Field(..., description="Career title")
    confidence: float = Field(..., description="Confidence score (percentage)")


class PredictionResponse(BaseModel):
    """Response model for career prediction"""
    careers: List[CareerRecommendation] = Field(..., description="Top career recommendations")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "careers": [
                    {"title": "Data Scientist", "confidence": 87.5},
                    {"title": "Product Manager", "confidence": 76.3},
                    {"title": "Business Analyst", "confidence": 65.2},
                    {"title": "UX Researcher", "confidence": 52.1},
                    {"title": "Software Engineer", "confidence": 48.9}
                ],
                "model_version": "1.0",
                "timestamp": "2025-10-07T20:20:02"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str]


# ==================== Helper Functions ====================

def load_model_and_dependencies():
    """Load trained model and initialize dependencies"""
    global model_package, confidence_scorer, feature_names
    
    model_path = "models/career_recommender_v1.pkl"
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        return False
    
    try:
        model_package = joblib.load(model_path)
        logger.info(f"✓ Model loaded: {model_package['model_name']} v{model_package['version']}")
        
        # Initialize confidence scorer
        confidence_scorer = ConfidenceScorer(model_package['career_names'])
        logger.info("✓ Confidence scorer initialized")
        
        # Define feature names (must match training)
        feature_names = [
            'analytical', 'creative', 'social',
            'education_encoded', 'experience', 'experience_normalized',
            'technical_skills_count', 'soft_skills_count', 'total_skills',
            'skill_diversity', 'tech_soft_ratio',
            'tech_oriented', 'creative_oriented', 'business_oriented',
            'social_oriented', 'interest_breadth',
            'personality_balance', 'analytical_creative_balance',
            'skills_education_score', 'exp_tech_score',
            'analytical_tech_alignment', 'creative_alignment',
            'social_skills_alignment',
            'career_readiness'
        ]
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def extract_features_from_request(request: PredictionRequest) -> Dict:
    """
    Extract and engineer features from API request
    
    Args:
        request: Prediction request object
        
    Returns:
        Dictionary of engineered features
    """
    # Define skill categories
    TECHNICAL_SKILLS = {
        'Python', 'Java', 'SQL', 'Machine Learning', 'Data Analysis',
        'Statistics', 'Cloud Computing', 'UI/UX', 'Excel', 'JavaScript'
    }
    
    SOFT_SKILLS = {
        'Communication', 'Leadership', 'Project Management', 'Business Strategy',
        'Creative Writing', 'Public Speaking', 'Negotiation'
    }
    
    TECH_INTERESTS = {'Technology', 'Science', 'Engineering'}
    CREATIVE_INTERESTS = {'Arts', 'Design', 'Creative', 'Media'}
    BUSINESS_INTERESTS = {'Business', 'Management', 'Finance'}
    SOCIAL_INTERESTS = {'Education', 'Health', 'Social'}
    
    # Education encoding
    education_mapping = {
        'High School': 1,
        'Bachelor': 2,
        'Master': 3,
        'PhD': 4
    }
    
    # Calculate feature values
    skills_set = set(request.skills)
    interests_set = set(request.interests)
    
    technical_skills_count = len(skills_set & TECHNICAL_SKILLS)
    soft_skills_count = len(skills_set & SOFT_SKILLS)
    total_skills = len(skills_set)
    
    tech_oriented = len(interests_set & TECH_INTERESTS)
    creative_oriented = len(interests_set & CREATIVE_INTERESTS)
    business_oriented = len(interests_set & BUSINESS_INTERESTS)
    social_oriented = len(interests_set & SOCIAL_INTERESTS)
    
    education_encoded = education_mapping[request.education]
    experience_normalized = min(request.experience / 20.0, 1.0)  # Normalize to [0, 1]
    
    # Calculate derived features
    skill_diversity = (technical_skills_count + soft_skills_count) / max(total_skills, 1)
    tech_soft_ratio = technical_skills_count / max(soft_skills_count, 1)
    interest_breadth = len(interests_set)
    
    personality_balance = np.std([
        request.personality.analytical,
        request.personality.creative,
        request.personality.social
    ])
    
    analytical_creative_balance = request.personality.analytical - request.personality.creative
    
    # Interaction features
    skills_education_score = total_skills * education_encoded
    exp_tech_score = experience_normalized * technical_skills_count
    analytical_tech_alignment = request.personality.analytical * technical_skills_count
    creative_alignment = request.personality.creative * creative_oriented
    social_skills_alignment = request.personality.social * soft_skills_count
    
    # Career readiness score
    career_readiness = (
        0.3 * (total_skills / 10) +
        0.25 * (education_encoded / 4) +
        0.25 * experience_normalized +
        0.2 * personality_balance
    )
    
    features = {
        'skills': request.skills,
        'analytical': request.personality.analytical,
        'creative': request.personality.creative,
        'social': request.personality.social,
        'education': request.education,
        'education_encoded': education_encoded,
        'experience': request.experience,
        'experience_normalized': experience_normalized,
        'technical_skills_count': technical_skills_count,
        'soft_skills_count': soft_skills_count,
        'total_skills': total_skills,
        'skill_diversity': skill_diversity,
        'tech_soft_ratio': tech_soft_ratio,
        'tech_oriented': tech_oriented,
        'creative_oriented': creative_oriented,
        'business_oriented': business_oriented,
        'social_oriented': social_oriented,
        'interest_breadth': interest_breadth,
        'personality_balance': personality_balance,
        'analytical_creative_balance': analytical_creative_balance,
        'skills_education_score': skills_education_score,
        'exp_tech_score': exp_tech_score,
        'analytical_tech_alignment': analytical_tech_alignment,
        'creative_alignment': creative_alignment,
        'social_skills_alignment': social_skills_alignment,
        'career_readiness': career_readiness
    }
    
    return features


# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Load model on API startup"""
    logger.info("Starting Career Recommendation API...")
    success = load_model_and_dependencies()
    if not success:
        logger.warning("API started without model - /predict endpoint will not work")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Career Recommendation Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_package is not None else "model_not_loaded",
        model_loaded=model_package is not None,
        model_version=model_package['version'] if model_package else None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_careers(request: PredictionRequest):
    """
    Predict top career recommendations with confidence scores
    
    Args:
        request: User profile with skills, interests, personality, education, and experience
        
    Returns:
        Top 5 career recommendations with confidence scores
    """
    # Check if model is loaded
    if model_package is None or confidence_scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure model file exists at models/career_recommender_v1.pkl"
        )
    
    try:
        # Extract features from request
        user_features = extract_features_from_request(request)
        
        # Debug: Check if feature_names is set
        if feature_names is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Feature names not initialized. Model may not be loaded correctly."
            )
        
        # Prepare feature vector for model
        X = np.array([[user_features[f] for f in feature_names]])
        
        # Get model predictions
        model = model_package['model']
        y_pred_proba = model.predict_proba(X)
        
        # Convert probabilities to array
        probabilities = np.array([proba[0][1] for proba in y_pred_proba])
        
        # Calculate confidence scores
        recommendations = confidence_scorer.calculate_confidence_scores(
            model_probabilities=probabilities,
            user_features=user_features,
            top_k=5
        )
        
        # Format response
        career_recommendations = [
            CareerRecommendation(title=rec['title'], confidence=rec['confidence'])
            for rec in recommendations
        ]
        
        response = PredictionResponse(
            careers=career_recommendations,
            model_version=model_package['version'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction successful: Top career = {career_recommendations[0].title}")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/careers", tags=["Information"])
async def get_available_careers():
    """Get list of all available careers"""
    if model_package is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "careers": model_package['career_names'],
        "count": len(model_package['career_names'])
    }


@app.get("/model-info", tags=["Information"])
async def get_model_info():
    """Get information about the loaded model"""
    if model_package is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_name": model_package['model_name'],
        "version": model_package['version'],
        "num_careers": len(model_package['career_names']),
        "feature_count": len(feature_names)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
