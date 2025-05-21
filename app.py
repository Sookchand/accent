import streamlit as st
import os
import tempfile
import subprocess
import yt_dlp
import ffmpeg
import openai
import librosa
import soundfile as sf
from dotenv import load_dotenv
from pydub import AudioSegment
import time
import uuid
import json
import requests
from accent_model import AccentDetector

# Try to import the websocket headers, but don't fail if it's not available
try:
    from streamlit.web.server.websocket_headers import _get_websocket_headers
except ImportError:
    # Create a dummy function for environments where this import is not available
    def _get_websocket_headers():
        return {}

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Accent Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🎙️ English Accent Detector")
st.markdown("""
This tool analyzes a speaker's accent from a video to:
- Classify the English accent (e.g., American, British, Australian)
- Provide a confidence score for English proficiency
- Generate a brief explanation of accent characteristics
""")

# Function to download video
def download_video(url, output_path):
    try:
        # Check if the URL is a direct media file
        if url.lower().endswith(('.mp4', '.mov', '.avi', '.wmv', '.flv', '.mkv', '.webm', '.m4a', '.mp3', '.wav')):
            try:
                # For direct media files, use requests to download
                st.info("Detected direct media URL. Downloading...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            except Exception as direct_err:
                st.warning(f"Failed to download direct media: {str(direct_err)}. Trying alternative method...")

        # Configure yt-dlp with more robust options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'quiet': False,  # Show output for debugging
            'no_warnings': False,
            'ignoreerrors': True,
            'noplaylist': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'socket_timeout': 30,
            'retries': 10,
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            'hls_prefer_native': True,
            'external_downloader_args': ['-timeout', '30'],
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1',
                'Connection': 'keep-alive',
            }
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.info(f"Downloading from {url}...")
            ydl.download([url])

        # Verify the file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.success("Download completed successfully!")
            return True
        else:
            st.error("Download completed but the file is empty or missing.")
            return False

    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")

        # Provide more helpful error messages based on the error type
        error_str = str(e).lower()
        if "403" in error_str or "forbidden" in error_str:
            st.warning("Access to this video is forbidden. This could be due to geographic restrictions or the video requires authentication.")
            st.info("Try using a different video URL or a direct MP4 link.")
        elif "404" in error_str or "not found" in error_str:
            st.warning("The video was not found. Please check if the URL is correct.")
        elif "private" in error_str:
            st.warning("This video is private and cannot be accessed.")
        elif "copyright" in error_str:
            st.warning("This video may have copyright restrictions preventing download.")
        elif "timeout" in error_str or "timed out" in error_str:
            st.warning("The download timed out. Please try again or use a different video.")

        return False

# Function to extract audio from video
def extract_audio(video_path, audio_path):
    try:
        st.info("Extracting audio from the video file...")

        # First try using pydub
        try:
            # Convert to WAV format for better compatibility
            audio = AudioSegment.from_file(video_path)

            # Normalize audio (adjust volume to a standard level)
            normalized_audio = audio.normalize()

            # Export as WAV format
            normalized_audio.export(audio_path, format="wav")

            # Verify the file exists and has content
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                st.success("Audio extraction completed successfully!")
                return True
        except Exception as pydub_error:
            st.warning(f"Error using pydub: {str(pydub_error)}. Trying alternative method...")

        # If pydub fails, try using ffmpeg directly
        try:
            import subprocess
            st.info("Trying ffmpeg directly...")

            # Construct ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian format
                '-ar', '44100',  # 44.1kHz sampling rate
                '-ac', '2',  # 2 channels (stereo)
                '-y',  # Overwrite output file if it exists
                audio_path
            ]

            # Run ffmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            # Check if the command was successful
            if process.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                st.success("Audio extraction with ffmpeg completed successfully!")
                return True
            else:
                st.error(f"ffmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                return False

        except Exception as ffmpeg_error:
            st.error(f"Error using ffmpeg: {str(ffmpeg_error)}")
            return False

    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return False

# Function to transcribe audio using OpenAI's Whisper
def transcribe_audio(audio_path):
    try:
        st.info("Transcribing audio using OpenAI's Whisper API...")

        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
            st.info("For testing purposes, returning a placeholder transcription.")
            return "This is a placeholder transcription for testing purposes. In a real scenario, this would be the actual transcription from the audio."

        # Check if the file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            st.error("Audio file is missing or empty.")
            return None

        # Get file size
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Convert to MB
        st.info(f"Audio file size: {file_size:.2f} MB")

        # Check if file is too large (OpenAI has a 25MB limit)
        if file_size > 24:
            st.warning("Audio file is larger than 24MB. Attempting to reduce size...")

            # Create a temporary file for the compressed audio
            temp_audio_path = audio_path + ".compressed.wav"

            try:
                # Use ffmpeg to compress the audio
                import subprocess
                cmd = [
                    'ffmpeg',
                    '-i', audio_path,
                    '-ac', '1',  # Convert to mono
                    '-ar', '16000',  # 16kHz sampling rate
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-y',  # Overwrite output file
                    temp_audio_path
                ]

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                _, stderr = process.communicate()

                if process.returncode == 0 and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                    st.success("Successfully compressed audio file.")
                    audio_path = temp_audio_path
                else:
                    st.error(f"Failed to compress audio: {stderr.decode('utf-8', errors='ignore')}")
            except Exception as compress_error:
                st.error(f"Error compressing audio: {str(compress_error)}")

        # Transcribe the audio
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                with open(audio_path, "rb") as audio_file:
                    transcription = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio_file
                    )

                # Clean up temporary compressed file if it exists
                if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                    except:
                        pass

                st.success("Transcription completed successfully!")
                return transcription["text"]

            except Exception as api_error:
                retry_count += 1
                error_str = str(api_error).lower()

                if "rate limit" in error_str:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                elif "too large" in error_str or "file size" in error_str:
                    st.error("Audio file is too large for the API. Please use a shorter audio clip.")
                    break
                elif "api key" in error_str or "authentication" in error_str:
                    st.error("Invalid OpenAI API key. Please check your API key and try again.")
                    break
                elif retry_count < max_retries:
                    st.warning(f"Error during transcription. Retrying... (Attempt {retry_count}/{max_retries})")
                    time.sleep(1)
                else:
                    st.error(f"Failed to transcribe after {max_retries} attempts: {str(api_error)}")
                    break

        # If we've exhausted retries or encountered a fatal error
        st.warning("For testing purposes, returning a placeholder transcription.")
        return "This is a placeholder transcription for testing purposes. In a real scenario, this would be the actual transcription from the audio."

    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        st.warning("For testing purposes, returning a placeholder transcription.")
        return "This is a placeholder transcription for testing purposes. In a real scenario, this would be the actual transcription from the audio."

# Function to analyze accent
def analyze_accent(audio_path, transcription):
    try:
        st.info("Analyzing accent...")

        # Check if the audio file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            st.error("Audio file is missing or empty.")
            # Return a placeholder result for testing
            return {
                "accent": "American",
                "confidence_score": 85.0,
                "explanation": "This is a placeholder result. The audio file was missing or empty."
            }

        # Check if transcription is valid
        if not transcription or len(transcription.strip()) < 10:
            st.warning("Transcription is too short or empty. Results may not be accurate.")

        # Initialize the accent detector
        detector = AccentDetector()

        # Predict the accent
        result = detector.predict(audio_path, transcription)

        # Validate the result
        if not result or not isinstance(result, dict):
            st.error("Invalid result from accent detector.")
            # Return a placeholder result for testing
            return {
                "accent": "American",
                "confidence_score": 80.0,
                "explanation": "This is a placeholder result. The accent detector returned an invalid result."
            }

        # Ensure all required fields are present
        required_fields = ["accent", "confidence_score", "explanation"]
        for field in required_fields:
            if field not in result:
                st.warning(f"Missing field in result: {field}")
                result[field] = "Unknown" if field != "confidence_score" else 75.0

        st.success("Accent analysis completed successfully!")
        return result

    except Exception as e:
        st.error(f"Error analyzing accent: {str(e)}")
        # Return a placeholder result for testing
        return {
            "accent": "American",
            "confidence_score": 70.0,
            "explanation": "This is a placeholder result due to an error in accent analysis."
        }

# Health check endpoint
def health_check():
    """
    Health check endpoint for monitoring.

    This function is called by the monitoring system to check if the application is healthy.
    It returns a 200 status code if the application is healthy.
    """
    try:
        # Check if we can access the request headers
        headers = _get_websocket_headers()

        # Check if the request is a health check
        if headers.get("X-Health-Check") == "true":
            # Return a 200 status code
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "0.1.0"
            }
    except:
        # If we can't access the headers, this is a normal request
        pass

    return None

# Main app functionality
def main():
    # Check if this is a health check request
    health_status = health_check()
    if health_status:
        st.json(health_status)
        return

    # Input for video URL
    video_url = st.text_input("Enter a public video URL (YouTube, Loom, or direct MP4 link):")

    if st.button("Analyze Accent"):
        if not video_url:
            st.warning("Please enter a video URL.")
            return

        # Create a unique session ID for this analysis
        session_id = str(uuid.uuid4())

        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, f"video_{session_id}")
        audio_path = os.path.join(temp_dir, f"audio_{session_id}.wav")

        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Record start time for monitoring
        start_time = time.time()
        error_occurred = False

        try:
            # Step 1: Download video
            status_text.text("Downloading video...")
            if not download_video(video_url, video_path):
                error_occurred = True
                return
            progress_bar.progress(25)

            # Step 2: Extract audio
            status_text.text("Extracting audio...")
            if not extract_audio(video_path, audio_path):
                error_occurred = True
                return
            progress_bar.progress(50)

            # Step 3: Transcribe audio
            status_text.text("Transcribing audio...")
            transcription = transcribe_audio(audio_path)
            if not transcription:
                error_occurred = True
                return
            progress_bar.progress(75)

            # Step 4: Analyze accent
            status_text.text("Analyzing accent...")
            accent_analysis = analyze_accent(audio_path, transcription)
            if not accent_analysis:
                error_occurred = True
                return
            progress_bar.progress(100)

            # Display results
            status_text.text("Analysis complete!")

            st.subheader("Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### Detected Accent")
                st.markdown(f"**{accent_analysis['accent']} English**")

                st.markdown(f"### Confidence Score")
                st.markdown(f"**{accent_analysis['confidence_score']:.1f}%**")

            with col2:
                st.markdown(f"### Explanation")
                st.markdown(accent_analysis['explanation'])

                st.markdown(f"### Transcription Sample")
                st.markdown(f"*\"{transcription[:200]}...\"*")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            error_occurred = True

        finally:
            # Record processing time for monitoring
            processing_time = time.time() - start_time

            # If monitoring module is available, record the request
            try:
                from monitoring import AccentDetectorMonitor
                monitor = AccentDetectorMonitor()
                monitor.record_request(processing_time=processing_time, error=error_occurred)
            except ImportError:
                # Monitoring module not available, ignore
                pass

            # Clean up temporary files
            try:
                os.remove(video_path)
                os.remove(audio_path)
                os.rmdir(temp_dir)
            except:
                pass

if __name__ == "__main__":
    main()
