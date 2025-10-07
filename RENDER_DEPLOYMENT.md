# Deploy to Render - Step by Step Guide

## 🚀 Quick Deploy (5 Minutes)

### Prerequisites
- GitHub account
- Render account (free): https://render.com

---

## Step 1: Prepare Your Code

### A. Train the Model First (IMPORTANT!)

Before deploying, you **must** train the model locally:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebooks to train model
jupyter notebook notebooks/01_data_engineering.ipynb
# Run all cells

jupyter notebook notebooks/02_model_development.ipynb
# Run all cells - this creates models/career_recommender_v1.pkl
```

**Verify model file exists:**
```bash
dir models\career_recommender_v1.pkl  # Windows
ls -la models/career_recommender_v1.pkl  # Mac/Linux
```

### B. Update .gitignore for Model File

The model file needs to be committed for Render deployment.

**Edit `.gitignore`** and comment out the model exclusion:

```gitignore
# Model files (large)
# *.pkl          <- Comment this line
# *.h5
# *.pt
# *.pth
# models/*.pkl   <- Comment this line
!models/.gitkeep
```

---

## Step 2: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit (including the model file)
git commit -m "Initial commit with trained model"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/career-recommender.git
git branch -M main
git push -u origin main
```

**⚠️ Important:** Make sure `models/career_recommender_v1.pkl` is included in the commit!

---

## Step 3: Deploy on Render

### Option A: Using render.yaml (Automated)

1. **Go to Render Dashboard**: https://dashboard.render.com

2. **Click "New +" → "Blueprint"**

3. **Connect your GitHub repository**
   - Authorize Render to access your GitHub
   - Select your repository: `career-recommender`

4. **Render will automatically detect `render.yaml`**
   - Service Name: `career-recommender-api`
   - Environment: Python
   - Plan: Free

5. **Click "Apply"**

6. **Wait for deployment** (~5-10 minutes)
   - Render will install dependencies
   - Start the API
   - Assign a public URL

### Option B: Manual Setup

1. **Go to Render Dashboard**: https://dashboard.render.com

2. **Click "New +" → "Web Service"**

3. **Connect Repository**
   - Connect your GitHub account
   - Select your repository

4. **Configure Service:**
   ```
   Name: career-recommender-api
   Environment: Python 3
   Region: Oregon (US West)
   Branch: main
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.api:app --host 0.0.0.0 --port $PORT
   Plan: Free
   ```

5. **Advanced Settings:**
   - Health Check Path: `/health`
   - Auto-Deploy: Yes

6. **Click "Create Web Service"**

---

## Step 4: Monitor Deployment

### Check Build Logs

In Render dashboard, you'll see:
```
==> Installing dependencies
Collecting pandas==2.0.3
Collecting scikit-learn==1.3.0
...
==> Build successful

==> Starting service
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:10000
```

### Your API URL

Once deployed, your API will be available at:
```
https://career-recommender-api.onrender.com
```

---

## Step 5: Test Your Deployment

### A. Health Check

```bash
curl https://career-recommender-api.onrender.com/health
```

### B. Make a Prediction

```bash
curl -X POST "https://career-recommender-api.onrender.com/predict" \
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

### C. View API Docs

Open in browser:
```
https://career-recommender-api.onrender.com/docs
```

---

## ⚠️ Important Notes for Free Tier

### Cold Starts
- Free tier spins down after 15 minutes of inactivity
- First request after inactivity takes ~30-60 seconds (cold start)
- Subsequent requests are fast

### Workaround for Cold Starts:
Use a service like **UptimeRobot** to ping your API every 14 minutes:
1. Sign up at https://uptimerobot.com
2. Add monitor: `https://your-api.onrender.com/health`
3. Check interval: 5 minutes

### Resource Limits
- Free tier: 512 MB RAM, 0.1 CPU
- Sufficient for this ML model
- If you get memory errors, upgrade to paid plan ($7/month)

---

## 🐛 Troubleshooting

### Issue 1: Model File Not Found

**Error**: `Model file not found at models/career_recommender_v1.pkl`

**Solution:**
1. Train model locally first
2. Uncomment `*.pkl` in `.gitignore`
3. Add and commit: `git add models/*.pkl && git commit -m "Add model"`
4. Push: `git push`
5. Redeploy on Render

### Issue 2: Build Failed - Out of Memory

**Error**: `Killed` during pip install

**Solution:**
Reduce package versions in `requirements.txt`:
```
scikit-learn==1.2.0  # Use older, smaller version
xgboost==1.7.0
```

### Issue 3: Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution:**
Make sure your start command uses the correct path:
```
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

### Issue 4: Port Binding Error

**Error**: `Address already in use`

**Solution:**
Use Render's `$PORT` environment variable (already configured in `render.yaml`)

---

## 🔒 Security Considerations

### Add API Authentication (Optional)

Edit `src/api.py` to add API key authentication:

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_careers(request: PredictionRequest):
    # ... existing code
```

Add environment variable in Render:
```
X_API_KEY=your-secret-key-here
```

---

## 📊 Monitor Your API

### Render Built-in Monitoring
- View logs in Render dashboard
- CPU/Memory usage graphs
- Request count and response times

### External Monitoring (Optional)
- **Uptime**: UptimeRobot, Pingdom
- **Logging**: Logtail, Papertrail
- **Errors**: Sentry

---

## 🎯 Next Steps After Deployment

1. ✅ Test all API endpoints
2. ✅ Share your API URL with others
3. ✅ Add custom domain (Render supports this)
4. ✅ Monitor performance and logs
5. ✅ Consider upgrading to paid plan for production

---

## 📱 Share Your API

Once deployed, share:
- **API URL**: `https://your-app.onrender.com`
- **Swagger Docs**: `https://your-app.onrender.com/docs`
- **Health Check**: `https://your-app.onrender.com/health`

---

## 💰 Cost Breakdown

### Free Tier (Included)
- 750 hours/month
- 512 MB RAM
- 0.1 CPU
- Custom `.onrender.com` subdomain

### Paid Plans (Optional)
- **Starter**: $7/month
  - No spin down
  - 1 GB RAM
  - 0.5 CPU
  
- **Standard**: $25/month
  - 2 GB RAM
  - 1 CPU
  - Priority support

For this project, **Free Tier is sufficient** for demonstration and testing!

---

## ✅ Deployment Checklist

- [ ] Model trained locally (`models/career_recommender_v1.pkl` exists)
- [ ] `.gitignore` updated to include model file
- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] Service deployed on Render
- [ ] Health check passes
- [ ] Test prediction successful
- [ ] API documentation accessible

---

## 🆘 Need Help?

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **Project Issues**: Check your repository issues tab

---

**Your API will be live at**: `https://career-recommender-api.onrender.com` 🎉
