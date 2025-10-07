"""Quick test for local API"""
import requests
import json

API_URL = "http://localhost:8000"

# Test prediction
profile = {
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

try:
    print("Testing local API at:", API_URL)
    
    # Health check
    health = requests.get(f"{API_URL}/health")
    print(f"\nHealth: {health.json()}")
    
    # Prediction
    print("\nMaking prediction...")
    response = requests.post(f"{API_URL}/predict", json=profile)
    
    if response.status_code == 200:
        data = response.json()
        print("\n[SUCCESS] Predictions:")
        for career in data['careers']:
            print(f"  - {career['title']}: {career['confidence']:.2f}%")
    else:
        print(f"\n[ERROR] {response.status_code}: {response.json()}")
        
except Exception as e:
    print(f"\n[ERROR] {e}")
