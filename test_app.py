import unittest
import os
import tempfile
import sys
from unittest.mock import patch, MagicMock, mock_open

# Add streamlit mock to avoid import errors
sys.modules['streamlit'] = MagicMock()
import streamlit as st

# Now import the functions from app
from app import download_video, extract_audio, transcribe_audio, analyze_accent

class TestApp(unittest.TestCase):
    """
    Test cases for the app functions.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "test_video")
        self.audio_path = os.path.join(self.temp_dir, "test_audio.wav")

        # Create dummy files
        with open(self.video_path, "w") as f:
            f.write("dummy video content")

        with open(self.audio_path, "w") as f:
            f.write("dummy audio content")

    def tearDown(self):
        """
        Clean up test environment.
        """
        # Remove temporary files
        try:
            os.remove(self.video_path)
            os.remove(self.audio_path)
            os.rmdir(self.temp_dir)
        except:
            pass

    @patch('yt_dlp.YoutubeDL')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_download_video_youtube(self, mock_getsize, mock_exists, mock_ydl):
        """
        Test the download_video function with a YouTube URL.
        """
        # Configure the mocks
        mock_ydl_instance = MagicMock()
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file

        # Call the function
        result = download_video("https://www.youtube.com/watch?v=test", self.video_path)

        # Assert the result
        self.assertTrue(result)
        mock_ydl_instance.download.assert_called_once_with(["https://www.youtube.com/watch?v=test"])

    @patch('requests.get')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_download_video_direct(self, mock_getsize, mock_exists, mock_requests_get):
        """
        Test the download_video function with a direct media URL.
        """
        # Configure the mocks
        mock_response = MagicMock()
        mock_requests_get.return_value = mock_response
        mock_response.iter_content.return_value = [b'test data']
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file

        # Mock the open function
        with patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            # Call the function
            result = download_video("https://example.com/video.mp4", self.video_path)

            # Assert the result
            self.assertTrue(result)
            mock_requests_get.assert_called_once_with("https://example.com/video.mp4", stream=True, timeout=30)
            mock_open.assert_called_once_with(self.video_path, 'wb')

    @patch('pydub.AudioSegment.from_file')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_extract_audio_pydub(self, mock_getsize, mock_exists, mock_audio_segment):
        """
        Test the extract_audio function using pydub.
        """
        # Configure the mocks
        mock_audio = MagicMock()
        mock_normalized_audio = MagicMock()
        mock_audio_segment.return_value = mock_audio
        mock_audio.normalize.return_value = mock_normalized_audio
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file

        # Call the function
        result = extract_audio(self.video_path, self.audio_path)

        # Assert the result
        self.assertTrue(result)
        mock_audio.normalize.assert_called_once()
        mock_normalized_audio.export.assert_called_once_with(self.audio_path, format="wav")

    @patch('pydub.AudioSegment.from_file')
    @patch('subprocess.Popen')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_extract_audio_ffmpeg_fallback(self, mock_getsize, mock_exists, mock_popen, mock_audio_segment):
        """
        Test the extract_audio function with ffmpeg fallback.
        """
        # Configure the mocks to make pydub fail
        mock_audio_segment.side_effect = Exception("Pydub error")

        # Configure subprocess mock
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"stdout", b"stderr")

        # Configure file checks
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file

        # Call the function
        result = extract_audio(self.video_path, self.audio_path)

        # Assert the result
        self.assertTrue(result)
        mock_popen.assert_called_once()
        mock_process.communicate.assert_called_once()

    @patch('openai.Audio.transcribe')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.getenv')
    def test_transcribe_audio_success(self, mock_getenv, mock_getsize, mock_exists, mock_transcribe):
        """
        Test the transcribe_audio function with successful API call.
        """
        # Configure the mocks
        mock_getenv.return_value = "fake-api-key"
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file
        mock_transcribe.return_value = {"text": "This is a test transcription."}

        # Call the function with a patch for open
        with patch('builtins.open', unittest.mock.mock_open(read_data=b'dummy audio data')):
            result = transcribe_audio(self.audio_path)

        # Assert the result
        self.assertEqual(result, "This is a test transcription.")
        mock_transcribe.assert_called_once()

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.getenv')
    def test_transcribe_audio_no_api_key(self, mock_getenv, mock_getsize, mock_exists):
        """
        Test the transcribe_audio function with no API key.
        """
        # Configure the mocks
        mock_getenv.return_value = None  # No API key
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file

        # Call the function
        result = transcribe_audio(self.audio_path)

        # Assert the result is a placeholder
        self.assertTrue(isinstance(result, str))
        self.assertTrue(len(result) > 0)
        self.assertTrue("placeholder" in result.lower())

    @patch('openai.Audio.transcribe')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.getenv')
    @patch('time.sleep')  # Mock sleep to avoid waiting in tests
    def test_transcribe_audio_retry(self, mock_sleep, mock_getenv, mock_getsize, mock_exists, mock_transcribe):
        """
        Test the transcribe_audio function with retry logic.
        """
        # We need to access mock_sleep to avoid the unused variable warning
        mock_sleep.return_value = None
        # Configure the mocks
        mock_getenv.return_value = "fake-api-key"
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file

        # Configure transcribe to fail once then succeed
        mock_transcribe.side_effect = [
            Exception("Rate limit exceeded"),
            {"text": "This is a test transcription."}
        ]

        # Call the function with a patch for open
        with patch('builtins.open', unittest.mock.mock_open(read_data=b'dummy audio data')):
            result = transcribe_audio(self.audio_path)

        # Assert the result
        self.assertEqual(result, "This is a test transcription.")
        self.assertEqual(mock_transcribe.call_count, 2)  # Called twice due to retry

    @patch('accent_model.AccentDetector')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_analyze_accent_success(self, mock_getsize, mock_exists, mock_detector_class):
        """
        Test the analyze_accent function with successful prediction.
        """
        # Configure the mocks
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_detector.predict.return_value = {
            "accent": "American",
            "confidence_score": 85.5,
            "explanation": "The speaker demonstrates typical American English pronunciation patterns."
        }

        # Call the function
        result = analyze_accent(self.audio_path, "This is a test transcription.")

        # Assert the result
        self.assertEqual(result["accent"], "American")
        self.assertEqual(result["confidence_score"], 85.5)
        self.assertEqual(result["explanation"], "The speaker demonstrates typical American English pronunciation patterns.")
        mock_detector.predict.assert_called_once_with(self.audio_path, "This is a test transcription.")

    @patch('os.path.exists')
    def test_analyze_accent_missing_file(self, mock_exists):
        """
        Test the analyze_accent function with missing audio file.
        """
        # Configure the mocks
        mock_exists.return_value = False  # File doesn't exist

        # Call the function
        result = analyze_accent(self.audio_path, "This is a test transcription.")

        # Assert the result is a placeholder
        self.assertEqual(result["accent"], "American")
        self.assertTrue(isinstance(result["confidence_score"], float))
        self.assertTrue("placeholder" in result["explanation"].lower())

    @patch('accent_model.AccentDetector')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_analyze_accent_invalid_result(self, mock_getsize, mock_exists, mock_detector_class):
        """
        Test the analyze_accent function with invalid result from detector.
        """
        # Configure the mocks
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_detector.predict.return_value = None  # Invalid result

        # Call the function
        result = analyze_accent(self.audio_path, "This is a test transcription.")

        # Assert the result is a placeholder
        self.assertEqual(result["accent"], "American")
        self.assertTrue(isinstance(result["confidence_score"], float))
        self.assertTrue("invalid result" in result["explanation"].lower())

if __name__ == "__main__":
    unittest.main()
