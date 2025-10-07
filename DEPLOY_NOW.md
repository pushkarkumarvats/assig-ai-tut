# 🚀 Quick Deploy to Render - 5 Steps

## Before You Start
**YOU MUST TRAIN THE MODEL FIRST!** The model file needs to exist before deployment.

---

## Step 1: Train Model Locally (Required!)

```bash
# Install dependencies
pip install -r requirements.txt

# Open and run ALL cells in these notebooks:
jupyter notebook notebooks/01_data_engineering.ipynb
jupyter notebook notebooks/02_model_development.ipynb
```

**Verify the model file exists:**
```bash
# Windows
dir models\career_recommender_v1.pkl

# Mac/Linux  
ls models/career_recommender_v1.pkl
```

✅ If you see the file, proceed to Step 2.

---

## Step 2: Push to GitHub

```bash
# Initialize git
git init

# Add all files (model is now included!)
git add .
git commit -m "Career Recommendation Engine with trained model"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/career-recommender.git
git branch -M main
git push -u origin main
```

---

## Step 3: Create Render Account

1. Go to: https://render.com
2. Click "Get Started for Free"
3. Sign up with GitHub

---

## Step 4: Deploy on Render

### Option A: Automatic (Uses render.yaml)

1. **Dashboard**: https://dashboard.render.com
2. Click **"New +"** → **"Blueprint"**
3. **Connect Repository**: Select your GitHub repo
4. Click **"Apply"**
5. ✅ Done! Wait 5-10 minutes for deployment

### Option B: Manual

1. **Dashboard**: https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. **Connect Repository**: Select your GitHub repo
4. **Configure**:
   - Name: `career-recommender-api`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
5. Click **"Create Web Service"**

---

## Step 5: Test Your API

Your API will be live at:
```
https://career-recommender-api.onrender.com
```

### Test in Browser
```
https://career-recommender-api.onrender.com/docs
```

### Test with cURL
```bash
curl https://career-recommender-api.onrender.com/health
```

---

## 🎉 You're Done!

Your API is now live and accessible to anyone on the internet!

**Share these URLs:**
- API: `https://career-recommender-api.onrender.com`
- Docs: `https://career-recommender-api.onrender.com/docs`

---

## ⚠️ Note on Free Tier

- API spins down after 15 min of inactivity
- First request after sleep = 30-60 seconds (cold start)
- After that, it's fast!

---

## Need Help?

Read the full guide: **RENDER_DEPLOYMENT.md**
