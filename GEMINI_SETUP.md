# Google Gemini API Setup Guide

This guide will help you set up Google Gemini API as an alternative to OpenAI for transcription in your accent detection application.

## Why Use Google Gemini API?

- **Free tier available**: Google offers generous free usage limits
- **High-quality transcription**: Excellent speech-to-text capabilities
- **Alternative to OpenAI**: Useful when OpenAI quota is exceeded
- **Multi-modal capabilities**: Can process audio, video, and text

## Getting Your Google Gemini API Key

### Step 1: Go to Google AI Studio

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account

### Step 2: Create an API Key

1. Click on "Create API Key"
2. Choose "Create API key in new project" (recommended for new users)
3. Copy the generated API key

### Step 3: Add the API Key to Your Environment

1. Open your `.env` file in the project directory
2. Replace `your_gemini_api_key_here` with your actual API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## How the Application Uses Gemini API

The application now supports multiple transcription APIs in order of preference:

1. **Google Gemini API** (if available and configured)
2. **OpenAI Whisper API** (if available and configured)
3. **Placeholder transcription** (for testing when no APIs are available)

## Benefits of Using Gemini API

### 1. Cost-Effective
- Generous free tier with high usage limits
- Competitive pricing for paid usage

### 2. High Quality
- Excellent transcription accuracy
- Support for multiple languages and accents

### 3. Reliability
- Google's robust infrastructure
- High availability and performance

### 4. Easy Integration
- Simple API interface
- Good documentation and support

## Usage Limits

### Free Tier
- 15 requests per minute
- 1,500 requests per day
- 1 million tokens per minute

### Paid Tier
- Higher rate limits
- Pay-per-use pricing
- Enterprise support available

## Testing Your Setup

1. Make sure you have installed the required package:
   ```bash
   pip install google-generativeai
   ```

2. Add your API key to the `.env` file

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Try analyzing a video or audio file. You should see:
   ```
   Attempting transcription with Google Gemini API...
   Transcription completed successfully with Google Gemini!
   ```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Make sure your `.env` file is in the project root directory
   - Verify the API key is correctly formatted without extra spaces

2. **Permission Denied**
   - Check that your API key is valid and active
   - Ensure you haven't exceeded your usage limits

3. **File Size Issues**
   - Gemini API has file size limits
   - The app will automatically compress large files

4. **Network Issues**
   - Check your internet connection
   - Verify that Google APIs are accessible from your network

### Error Messages

- **"Google Gemini API is not available"**: Install the `google-generativeai` package
- **"Google Gemini API key is not set"**: Add your API key to the `.env` file
- **"Error with Gemini API"**: Check your API key and usage limits

## Comparison: Gemini vs OpenAI

| Feature | Google Gemini | OpenAI Whisper |
|---------|---------------|----------------|
| Free Tier | Generous limits | Limited credits |
| Accuracy | Excellent | Excellent |
| Speed | Fast | Fast |
| File Size Limit | Large files supported | 25MB limit |
| Languages | Multi-language | Multi-language |
| Cost | Cost-effective | Higher cost |

## Next Steps

1. **Get your API key** from Google AI Studio
2. **Add it to your `.env` file**
3. **Test the transcription** with your accent detection app
4. **Enjoy improved confidence scores** with accurate transcriptions

For more information, visit the [Google AI documentation](https://ai.google.dev/docs).
