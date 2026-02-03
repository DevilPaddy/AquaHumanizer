# 🚀 AquaHumanizer Deployment Guide

## Quick Deploy (5 minutes)

### Step 1: Deploy HF Space (Model Service)

1. **Go to**: https://huggingface.co/spaces
2. **Create New Space**:
   - Name: `aquahumanizer-model` (or your choice)
   - SDK: **Docker**
   - Hardware: **CPU Basic** (free)
3. **Upload Files**:
   - `app.py`
   - `requirements.txt`
   - `Dockerfile`
4. **Wait for Build** (5-10 minutes)
5. **Copy URL**: `https://your-username-aquahumanizer-model.hf.space`

### Step 2: Deploy Frontend to Vercel

1. **Push to GitHub**:
   ```bash
   # Upload the 'frontend/' folder contents to a new GitHub repo
   ```

2. **Deploy on Vercel**:
   - Go to: https://vercel.com/
   - **New Project** → **Import Git Repository**
   - Select your repository
   - **Framework**: Next.js (auto-detected)
   - **Root Directory**: Leave empty (or set to `frontend` if you uploaded the whole project)

3. **Set Environment Variable**:
   - Go to **Settings** → **Environment Variables**
   - Add: `HF_SPACE_URL` = `https://your-hf-space-url.hf.space`

4. **Deploy**: Vercel will automatically build and deploy

### Step 3: Test

1. **Visit your app**: `https://your-app.vercel.app`
2. **Test API**: `https://your-app.vercel.app/api/health`
3. **Humanize text**: Use the frontend form

## ✅ Success Criteria

- ✅ HF Space shows "Running" status
- ✅ Health API returns `{"status": "healthy"}`
- ✅ Frontend loads and accepts text input
- ✅ Text humanization works (may take 10-30s first time)

## 🔧 Troubleshooting

### HF Space Issues
- **Build fails**: Check Dockerfile syntax
- **Model not loading**: Wait longer, check logs
- **Timeout**: First request takes time (model loading)

### Vercel Issues
- **Build fails**: Check Node.js version (18+)
- **API errors**: Verify `HF_SPACE_URL` environment variable
- **CORS errors**: Should work automatically with same-origin requests

### Performance
- **First request slow**: Model needs to load (10-30s)
- **Subsequent requests**: Should be 2-5s
- **Cached requests**: <1s

## 💰 Cost

**100% Free** on free tiers:
- **HF Spaces**: Free CPU tier
- **Vercel**: Free hobby plan
- **GitHub**: Free public repositories

---

**Total deployment time: ~10 minutes**
**Ongoing maintenance: Zero**