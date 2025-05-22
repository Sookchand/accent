# English Accent Detector

A tool that analyzes a speaker's accent from a video or audio file to help evaluate English language proficiency for hiring purposes.

## Features

- **Multiple Input Methods**: Process YouTube URLs, direct media links, uploaded files, or local files
- **Audio Extraction**: Extracts audio from video for analysis
- **Accent Classification**: Identifies the speaker's English accent (e.g., British, American, Australian)
- **Confidence Scoring**: Provides a confidence score for English proficiency (0-100%)
- **Explanation**: Generates a brief explanation of accent characteristics
- **Alternative Suggestions**: Shows alternative accent possibilities when confidence is low

## Demo

You can try the live demo at: [Accent Detector App](https://accent-detector.streamlit.app/) (Note: This is a placeholder link)

## How It Works

1. **Input Processing**: The tool accepts YouTube URLs, direct media links, uploaded files, or local files
2. **Audio Extraction**: It extracts the audio track from the video or processes the audio file directly
3. **Transcription**: OpenAI's Whisper API transcribes the speech
4. **Accent Analysis**: A machine learning model analyzes the audio and transcription
5. **Results**: The tool displays the detected accent, confidence score, and explanation

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- OpenAI API key for Whisper transcription

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/accent-detector.git
   cd accent-detector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Install FFmpeg:
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`

### Running the App

Run the Streamlit app locally:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Usage

### Method 1: URL Input
1. Select the "URL Input" tab
2. Enter a public video URL (YouTube, Loom, or direct media link)
3. Click "Analyze URL"
4. Wait for the analysis to complete
5. View the results showing the detected accent, confidence score, and explanation

### Method 2: File Upload
1. Select the "File Upload" tab
2. Upload an audio or video file (MP4, MP3, WAV, etc.)
3. Click "Analyze File"
4. Wait for the analysis to complete
5. View the results

### Method 3: Local Sample
1. Select the "Local Sample" tab
2. Enter the path to a local audio or video file (e.g., "MP4_sample")
3. Click "Analyze Local File"
4. Wait for the analysis to complete
5. View the results
## Technical Details


- **Video Processing**: Uses yt-dlp to download videos from various platforms
- **Audio Extraction**: Uses ffmpeg and pydub for audio processing
- **Speech Recognition**: Uses OpenAI's Whisper for high-quality transcription
- **Accent Classification**: Currently uses a placeholder model (to be replaced with a trained ML model)

## Deployment Options

This project includes several deployment options:

1. **Docker**: Use `docker-compose up` to run the application in a container
2. **Kubernetes**: Configuration files in the `kubernetes/` directory
3. **AWS**: Terraform configuration in the `terraform/` directory
4. **Heroku**: Procfile and runtime.txt for Heroku deployment
5. **Google Cloud**: Deployment instructions in DEPLOYMENT.md
6. **Azure**: Deployment instructions in DEPLOYMENT.md

For detailed deployment instructions, see the [DEPLOYMENT.md](DEPLOYMENT.md) file.

## Limitations

- The current implementation uses a placeholder for accent detection
- Processing time depends on video length and server load
- Only supports English accent detection

## Advanced Model Development

For detailed guidance on developing a sophisticated accent detection model, refer to the [ADVANCED_MODEL.md](ADVANCED_MODEL.md) file.

### Training a Custom Model

The application includes several scripts to help you develop and train a custom accent detection model:

1. **collect_dataset.py**: Collect a dataset for training from YouTube videos or existing audio files
   ```
   python collect_dataset.py --output_dir dataset --source youtube --urls urls.txt --accent american
   ```

2. **create_sample_dataset.py**: Create a synthetic sample dataset for testing
   ```
   python create_sample_dataset.py --output_dir dataset/sample --num_samples 5
   ```

3. **train_model.py**: Train a new accent detection model on your dataset
   ```
   python train_model.py --data_dir dataset --output_dir models
   ```

4. **evaluate_model.py**: Evaluate a trained model on a test dataset
   ```
   python evaluate_model.py --model_path models/accent_model.pkl --test_dir dataset/test
   ```

The trained model will be automatically loaded by the application if it's placed in the `models` directory.

## Future Improvements

- Implement a proper machine learning model for accent detection (see [ADVANCED_MODEL.md](ADVANCED_MODEL.md))
- Add support for more languages and accents
- Improve processing speed and efficiency
- Add batch processing for multiple videos

## License

MIT
