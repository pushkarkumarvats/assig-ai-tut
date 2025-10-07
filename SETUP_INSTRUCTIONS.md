# Setup Instructions
## Career Recommendation Engine

Follow these steps to set up and run the Career Recommendation Engine on your local machine.

---

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **Git**: For cloning the repository
- **Virtual Environment**: Recommended

---

## Step 1: Clone the Repository (if applicable)

```bash
git clone <repository-url>
cd assig-ai-tut
```

---

## Step 2: Create Virtual Environment

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected Installation Time**: 2-5 minutes

---

## Step 4: Verify Data Files

Ensure the dataset is present:

```bash
# Check if data file exists
dir data\synthetic_user_profiles_large.csv  # Windows
ls data/synthetic_user_profiles_large.csv   # Linux/Mac
```

If missing, ensure `data/synthetic_user_profiles_large.csv` is in the correct location.

---

## Step 5: Run Data Preprocessing & Feature Engineering

### Option A: Using Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/01_data_engineering.ipynb
```

Run all cells in the notebook to:
- Load and analyze data
- Engineer features
- Create visualizations
- Save processed data

### Option B: Using Python Scripts

```bash
# Run preprocessing
python src/preprocessing.py

# Run feature engineering
python src/feature_engineering.py
```

---

## Step 6: Train Models

Open and run the model development notebook:

```bash
jupyter notebook notebooks/02_model_development.ipynb
```

Run all cells to:
- Train Random Forest and XGBoost models
- Compare model performance
- Implement confidence scoring
- Save trained model

**Expected Training Time**: 5-15 minutes (depends on dataset size and hardware)

---

## Step 7: Verify Model Files

Check that the model was saved successfully:

```bash
dir models\career_recommender_v1.pkl  # Windows
ls models/career_recommender_v1.pkl   # Linux/Mac
```

If the model file doesn't exist, re-run the model training notebook.

---

## Step 8: Start the API Server

### Option A: Using the run script (Recommended)

```bash
python run_api.py
```

### Option B: Using uvicorn directly

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:
```
Starting Career Recommendation Engine API
=========================================================
API will be available at:
  - Main API: http://localhost:8000
  - Swagger UI: http://localhost:8000/docs
  - ReDoc: http://localhost:8000/redoc
```

---

## Step 9: Test the API

### Option A: Using the test script

Open a **new terminal** (keep the API server running) and run:

```bash
python test_prediction.py
```

This will test the API with 5 different sample profiles.

### Option B: Using Swagger UI

1. Open browser: http://localhost:8000/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Modify the example request
5. Click "Execute"

### Option C: Using cURL

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"skills\": [\"Python\", \"Communication\"], \"interests\": [\"Technology\"], \"personality\": {\"analytical\": 0.8, \"creative\": 0.4, \"social\": 0.7}, \"education\": \"Bachelor\", \"experience\": 3}"
```

---

## Step 10: Run Tests

```bash
pytest tests/ -v
```

This will run the comprehensive test suite covering all API endpoints.

---

## Project Structure After Setup

```
assig-ai-tut/
├── data/
│   ├── synthetic_user_profiles_large.csv  [Original dataset]
│   └── processed/
│       ├── features.csv                    [Processed features]
│       ├── targets.npy                     [Target labels]
│       └── metadata.pkl                    [Metadata]
├── models/
│   ├── career_recommender_v1.pkl           [Trained model]
│   └── model_artifacts_v1.pkl              [Model metadata]
├── notebooks/
│   ├── 01_data_engineering.ipynb           [Data preprocessing]
│   └── 02_model_development.ipynb          [Model training]
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── confidence_scorer.py
│   └── api.py
├── tests/
│   └── test_api.py
└── requirements.txt
```

---

## Troubleshooting

### Issue 1: Module not found errors

**Solution**: Make sure you're in the correct directory and virtual environment is activated.

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Verify Python path
python -c "import sys; print(sys.executable)"
```

### Issue 2: Model not loaded error when calling API

**Error**: `503 Service Unavailable: Model not loaded`

**Solution**: 
1. Ensure you ran the model training notebook completely
2. Check that `models/career_recommender_v1.pkl` exists
3. Restart the API server

### Issue 3: Memory errors during training

**Solution**: 
- Reduce dataset size in preprocessing
- Use fewer trees in Random Forest: `n_estimators=50`
- Close other applications to free memory

### Issue 4: Port 8000 already in use

**Error**: `Address already in use`

**Solution**:

**Windows**:
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Linux/Mac**:
```bash
lsof -ti:8000 | xargs kill -9
```

Or use a different port:
```bash
uvicorn src.api:app --reload --port 8001
```

### Issue 5: Jupyter notebook kernel issues

**Solution**:
```bash
python -m ipykernel install --user --name=venv
```

Then select the 'venv' kernel in Jupyter.

---

## Optional: Docker Setup

### Build Docker Image

```bash
docker build -t career-recommender:1.0 .
```

### Run Docker Container

```bash
docker run -p 8000:8000 career-recommender:1.0
```

**Note**: Ensure `models/career_recommender_v1.pkl` exists before building the image.

---

## Quick Start Summary

For the impatient, here's the minimal command sequence:

```bash
# 1. Setup
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# 2. Train model (run notebooks or wait for pre-trained model)
jupyter notebook notebooks/01_data_engineering.ipynb
jupyter notebook notebooks/02_model_development.ipynb

# 3. Start API
python run_api.py

# 4. Test (in new terminal)
python test_prediction.py
```

---

## Next Steps

1. ✅ Explore the Swagger UI: http://localhost:8000/docs
2. ✅ Review the notebooks for detailed analysis
3. ✅ Read `DESIGN_DECISIONS.md` for implementation details
4. ✅ Review `API_DOCUMENTATION.md` for API usage
5. ✅ Run the test suite to verify everything works

---

## Support

If you encounter issues not covered in troubleshooting:

1. Check error logs in the terminal
2. Review the source code comments
3. Examine the test cases in `tests/test_api.py`
4. Verify all dependencies are installed correctly

---

## Performance Benchmarks

Expected performance on modern hardware:

| Task | Time |
|------|------|
| Data preprocessing | ~10-30 seconds |
| Feature engineering | ~5-15 seconds |
| Model training (RF) | ~2-5 minutes |
| Model training (XGB) | ~3-8 minutes |
| API startup | ~1-2 seconds |
| Single prediction | ~100-500 ms |

---

Happy coding! 🚀
