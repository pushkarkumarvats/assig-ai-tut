# Career Recommendation Engine - API Documentation

## Overview

The Career Recommendation Engine API provides AI-powered career recommendations based on user skills, interests, personality traits, education, and experience. The API uses a multi-label classification model with advanced confidence scoring to deliver personalized career suggestions.

**Base URL**: `http://localhost:8000`

**API Version**: 1.0.0

---

## Authentication

Currently, the API does not require authentication. This can be added for production deployments.

---

## Endpoints

### 1. Root Endpoint

**GET** `/`

Returns basic API information.

#### Response

```json
{
  "message": "Career Recommendation Engine API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### 2. Health Check

**GET** `/health`

Checks API health status and model availability.

#### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0"
}
```

#### Status Codes

- `200 OK`: API is healthy

---

### 3. Career Prediction

**POST** `/predict`

Predicts top career recommendations with confidence scores based on user profile.

#### Request Body

```json
{
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
```

#### Request Parameters

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `skills` | `array[string]` | Yes | List of user skills | Min 1 item |
| `interests` | `array[string]` | Yes | List of user interests | Min 1 item |
| `personality` | `object` | Yes | Personality trait scores | See below |
| `personality.analytical` | `float` | Yes | Analytical trait score | 0.0 - 1.0 |
| `personality.creative` | `float` | Yes | Creative trait score | 0.0 - 1.0 |
| `personality.social` | `float` | Yes | Social trait score | 0.0 - 1.0 |
| `education` | `string` | Yes | Education level | High School, Bachelor, Master, PhD |
| `experience` | `integer` | Yes | Years of work experience | 0 - 50 |

#### Response

```json
{
  "careers": [
    {
      "title": "Data Scientist",
      "confidence": 87.5
    },
    {
      "title": "Product Manager",
      "confidence": 76.3
    },
    {
      "title": "Business Analyst",
      "confidence": 65.2
    },
    {
      "title": "UX Researcher",
      "confidence": 52.1
    },
    {
      "title": "Software Engineer",
      "confidence": 48.9
    }
  ],
  "model_version": "1.0",
  "timestamp": "2025-10-07T20:20:02.123456"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `careers` | `array[object]` | Top 5 career recommendations |
| `careers[].title` | `string` | Career title |
| `careers[].confidence` | `float` | Confidence score (percentage) |
| `model_version` | `string` | Model version used for prediction |
| `timestamp` | `string` | Prediction timestamp (ISO 8601) |

#### Status Codes

- `200 OK`: Prediction successful
- `422 Unprocessable Entity`: Invalid input parameters
- `500 Internal Server Error`: Prediction failed
- `503 Service Unavailable`: Model not loaded

---

### 4. Get Available Careers

**GET** `/careers`

Returns list of all careers the model can predict.

#### Response

```json
{
  "careers": [
    "Data Scientist",
    "Software Engineer",
    "Business Analyst",
    "Product Manager",
    "UX Designer",
    "Marketing Specialist",
    "Financial Analyst",
    "Research Scientist"
  ],
  "count": 8
}
```

#### Status Codes

- `200 OK`: Careers retrieved successfully
- `503 Service Unavailable`: Model not loaded

---

### 5. Get Model Information

**GET** `/model-info`

Returns information about the loaded model.

#### Response

```json
{
  "model_name": "random_forest",
  "version": "1.0",
  "num_careers": 8,
  "feature_count": 24
}
```

#### Status Codes

- `200 OK`: Model info retrieved successfully
- `503 Service Unavailable`: Model not loaded

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Scenarios

#### 1. Invalid Education Level

**Request:**
```json
{
  "education": "Elementary"
}
```

**Response (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "education"],
      "msg": "Education must be one of: High School, Bachelor, Master, PhD",
      "type": "value_error"
    }
  ]
}
```

#### 2. Out of Range Personality Score

**Request:**
```json
{
  "personality": {
    "analytical": 1.5,
    "creative": 0.4,
    "social": 0.7
  }
}
```

**Response (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "personality", "analytical"],
      "msg": "ensure this value is less than or equal to 1.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

#### 3. Missing Required Field

**Request:**
```json
{
  "skills": ["Python"],
  "interests": ["Technology"]
}
```

**Response (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "personality"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Usage Examples

### Python

```python
import requests

url = "http://localhost:8000/predict"

payload = {
    "skills": ["Python", "Machine Learning", "Communication"],
    "interests": ["Technology", "Science"],
    "personality": {
        "analytical": 0.9,
        "creative": 0.3,
        "social": 0.6
    },
    "education": "Master",
    "experience": 5
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    print("Top Career Recommendations:")
    for career in data["careers"]:
        print(f"  - {career['title']}: {career['confidence']:.2f}%")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["Python", "Communication"],
    "interests": ["Technology", "Management"],
    "personality": {
      "analytical": 0.8,
      "creative": 0.4,
      "social": 0.7
    },
    "education": "Bachelor",
    "experience": 3
  }'
```

### JavaScript (Fetch)

```javascript
const url = "http://localhost:8000/predict";

const payload = {
  skills: ["UI/UX", "Creative Writing"],
  interests: ["Arts", "Design"],
  personality: {
    analytical: 0.3,
    creative: 0.9,
    social: 0.7
  },
  education: "Bachelor",
  experience: 2
};

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => {
    console.log("Career Recommendations:", data.careers);
  })
  .catch(error => {
    console.error("Error:", error);
  });
```

---

## Interactive Documentation

The API provides interactive documentation using Swagger UI and ReDoc:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- Explore all endpoints
- View detailed request/response schemas
- Test the API directly from your browser
- Download OpenAPI specification

---

## Rate Limiting

Currently, no rate limiting is implemented. For production deployment, consider implementing rate limiting based on:
- IP address
- API key (if authentication is added)
- User account

---

## Performance

### Response Times

- **/predict**: ~100-500ms (depends on model complexity)
- **/health**: <10ms
- **/careers**: <10ms
- **/model-info**: <10ms

### Optimization Tips

1. **Caching**: Cache frequently requested predictions
2. **Batch Predictions**: Implement batch endpoint for multiple users
3. **Model Optimization**: Use model quantization or ONNX runtime
4. **Load Balancing**: Deploy multiple instances behind load balancer

---

## Deployment

### Docker

```bash
# Build image
docker build -t career-recommender:1.0 .

# Run container
docker run -p 8000:8000 career-recommender:1.0
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn src.api:app --reload --port 8000
```

---

## Support and Contact

For issues, questions, or feature requests:
- Create an issue in the project repository
- Review the notebooks for implementation details
- Check test cases in `tests/test_api.py`

---

## Changelog

### Version 1.0.0 (2025-10-07)

- Initial release
- Multi-label career prediction
- Confidence scoring system
- 8 career categories supported
- Comprehensive input validation
- Interactive API documentation
