# Streamlit Cloud Deployment Guide

This guide will help you deploy your English Accent Detector to Streamlit Cloud.

## Prerequisites

1. **GitHub Account**: Your code must be in a GitHub repository
2. **API Keys**: At least one of the following:
   - Google Gemini API key (recommended)
   - OpenAI API key (alternative)

## Step 1: Prepare Your Repository

### 1.1 Push Your Code to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Add accent detection app with Gemini API support"

# Add remote repository
git remote add origin https://github.com/yourusername/accent-detector.git

# Push to GitHub
git push -u origin main
```

### 1.2 Verify Required Files

Make sure these files are in your repository:
- ✅ `app.py` (main application)
- ✅ `requirements.txt` (dependencies)
- ✅ `packages.txt` (system packages - contains `ffmpeg`)
- ✅ `.streamlit/config.toml` (Streamlit configuration)
- ✅ `accent_model.py` (accent detection model)

## Step 2: Get API Keys

### Option 1: Google Gemini API (Recommended)

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Choose "Create API key in new project"
5. Copy the generated API key

**Benefits:**
- Free tier with generous limits
- High-quality transcription
- Better performance than OpenAI for this use case

### Option 2: OpenAI API (Alternative)

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in to your account
3. Click "Create new secret key"
4. Copy the generated API key

**Note:** OpenAI requires billing setup after free credits are exhausted.

## Step 3: Deploy to Streamlit Cloud

### 3.1 Access Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"

### 3.2 Configure Your App

1. **Repository**: Select your GitHub repository
2. **Branch**: Choose `main` (or your default branch)
3. **Main file path**: Enter `app.py`
4. **App URL**: Choose a custom URL (optional)

### 3.3 Add Secrets (API Keys)

1. Click "Advanced settings"
2. Go to the "Secrets" section
3. Add your API key(s):

```toml
# For Google Gemini API (recommended)
GEMINI_API_KEY = "your_actual_gemini_api_key_here"

# For OpenAI API (optional fallback)
OPENAI_API_KEY = "your_actual_openai_api_key_here"
```

**Important:** Replace the placeholder values with your actual API keys.

### 3.4 Deploy

1. Click "Deploy!"
2. Wait for the deployment to complete (usually 2-5 minutes)
3. Your app will be available at the provided URL

## Step 4: Verify Deployment

### 4.1 Check API Status

1. Open your deployed app
2. Look at the sidebar for "🔑 API Status"
3. Verify that your API keys are recognized:
   - ✅ Green checkmark = API ready
   - ⚠️ Yellow warning = API key missing
   - ❌ Red X = API not available

### 4.2 Test Functionality

1. **Test File Upload**: Upload a small audio/video file
2. **Test URL Input**: Try a YouTube URL
3. **Check Transcription**: Verify that real transcription (not placeholder) is generated
4. **Verify Results**: Ensure accent detection provides reasonable confidence scores

## Step 5: Monitor and Maintain

### 5.1 Monitor Usage

- Check your API usage in Google AI Studio or OpenAI dashboard
- Monitor app performance in Streamlit Cloud dashboard

### 5.2 Update Your App

To update your deployed app:

```bash
# Make changes to your code
git add .
git commit -m "Update app with new features"
git push origin main
```

Streamlit Cloud will automatically redeploy when you push changes.

## Troubleshooting

### Common Issues

1. **"No API keys configured" error**
   - Check that secrets are properly added in Streamlit Cloud
   - Verify API key format (no extra spaces or quotes)

2. **"FFmpeg not found" error**
   - Ensure `packages.txt` contains `ffmpeg`
   - Check that the file is in the repository root

3. **Import errors**
   - Verify all dependencies are in `requirements.txt`
   - Check for version conflicts

4. **File upload issues**
   - File size limit is 200MB on Streamlit Cloud
   - Supported formats: MP4, MP3, WAV, etc.

### Getting Help

1. Check Streamlit Cloud logs in the app dashboard
2. Review the [Streamlit documentation](https://docs.streamlit.io/)
3. Check the [Google AI documentation](https://ai.google.dev/docs)

## Expected Performance

After successful deployment, you should see:

- **High confidence scores**: 70-90% (vs previous 30-40%)
- **Real transcriptions**: Actual speech-to-text instead of placeholders
- **Multiple input methods**: URL, file upload, and local files
- **Fast processing**: Efficient audio processing with FFmpeg

## Security Notes

1. **Never commit API keys** to your repository
2. **Use Streamlit secrets** for sensitive information
3. **Monitor API usage** to avoid unexpected charges
4. **Rotate API keys** periodically for security

## Next Steps

After successful deployment:

1. **Share your app** with users
2. **Collect feedback** on accent detection accuracy
3. **Monitor usage patterns** and optimize accordingly
4. **Consider upgrading** to paid API tiers for higher usage

Your accent detection app is now live and ready to use! 🎉
