#!/usr/bin/env python3
"""
Script to collect a real dataset for accent detection.

This script helps you collect audio samples from various sources to build a dataset
for training an accent detection model. It provides a simple interface to:
1. Download audio from YouTube videos with known accents
2. Process existing audio files
3. Record audio directly from your microphone

Usage:
    python collect_real_dataset.py

For detailed guidance on developing a sophisticated accent detection model,
refer to the ADVANCED_MODEL.md file.
"""

import os
import sys
import subprocess
import webbrowser
import time
import random
import string
import json
from datetime import datetime

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    # Check for FFmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        missing_deps.append("FFmpeg")
    
    # Check for pytube
    try:
        import pytube
    except ImportError:
        missing_deps.append("pytube")
    
    # Check for yt-dlp
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        missing_deps.append("yt-dlp")
    
    return missing_deps

def install_dependencies(missing_deps):
    """Provide instructions to install missing dependencies."""
    if "FFmpeg" in missing_deps:
        print("\n❌ FFmpeg is not installed or not in PATH.")
        print("Please run 'python install_ffmpeg.py' for installation instructions.")
    
    if "pytube" in missing_deps:
        print("\n❌ pytube is not installed.")
        print("Install it with: pip install pytube")
    
    if "yt-dlp" in missing_deps:
        print("\n❌ yt-dlp is not installed.")
        print("Install it with: pip install yt-dlp")
    
    if missing_deps:
        print("\nPlease install the missing dependencies and run this script again.")
        sys.exit(1)

def create_dataset_structure():
    """Create the dataset directory structure."""
    accents = ["american", "british", "australian", "indian", "canadian"]
    dataset_dir = "dataset"
    
    # Create main dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Create subdirectories for each accent
    for accent in accents:
        accent_dir = os.path.join(dataset_dir, accent)
        if not os.path.exists(accent_dir):
            os.makedirs(accent_dir)
    
    return dataset_dir, accents

def random_string(length=8):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def download_from_youtube(url, output_dir, accent):
    """
    Download audio from a YouTube video.
    
    Args:
        url (str): YouTube URL.
        output_dir (str): Directory to save the audio.
        accent (str): Accent label.
        
    Returns:
        str: Path to the downloaded audio file, or None if download failed.
    """
    print(f"Downloading from {url}...")
    
    # Generate temporary file paths
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    video_path = os.path.join(temp_dir, f"video_{random_string()}")
    audio_path = os.path.join(output_dir, accent, f"{accent}_{random_string()}.wav")
    
    # Try downloading with pytube
    try:
        import pytube
        
        print("Downloading with pytube...")
        yt = pytube.YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        
        if not stream:
            print("No suitable stream found. Trying audio-only stream...")
            stream = yt.streams.filter(only_audio=True).first()
        
        if not stream:
            print("No suitable stream found with pytube.")
            raise Exception("No suitable stream found")
        
        # Download the video
        stream.download(output_path=os.path.dirname(video_path), filename=os.path.basename(video_path))
        
        # Extract audio using FFmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian audio codec
            '-ar', '16000',  # 16 kHz sample rate
            '-ac', '1',  # Mono
            audio_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the audio file was created successfully
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print(f"Successfully downloaded and extracted audio to {audio_path}")
            return audio_path
        else:
            print("Failed to extract audio with FFmpeg.")
            
    except Exception as e:
        print(f"Error with pytube: {str(e)}")
    
    # If pytube fails, try yt-dlp
    try:
        print("Downloading with yt-dlp...")
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--output', audio_path,
            url
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the audio file was created successfully
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print(f"Successfully downloaded audio to {audio_path}")
            return audio_path
        else:
            print("Failed to download with yt-dlp.")
    
    except Exception as e:
        print(f"Error with yt-dlp: {str(e)}")
    
    print("Failed to download audio from YouTube.")
    return None

def process_existing_audio(file_path, output_dir, accent):
    """
    Process an existing audio file.
    
    Args:
        file_path (str): Path to the audio file.
        output_dir (str): Directory to save the processed audio.
        accent (str): Accent label.
        
    Returns:
        str: Path to the processed audio file, or None if processing failed.
    """
    print(f"Processing {file_path}...")
    
    # Generate output path
    output_path = os.path.join(output_dir, accent, f"{accent}_{random_string()}.wav")
    
    # Process audio using FFmpeg
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', file_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian audio codec
            '-ar', '16000',  # 16 kHz sample rate
            '-ac', '1',  # Mono
            output_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the audio file was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully processed audio to {output_path}")
            return output_path
        else:
            print("Failed to process audio with FFmpeg.")
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
    
    print("Failed to process audio file.")
    return None

def record_audio(output_dir, accent, duration=5):
    """
    Record audio from the microphone.
    
    Args:
        output_dir (str): Directory to save the recorded audio.
        accent (str): Accent label.
        duration (int): Duration of the recording in seconds.
        
    Returns:
        str: Path to the recorded audio file, or None if recording failed.
    """
    print(f"Recording {duration} seconds of audio...")
    
    # Generate output path
    output_path = os.path.join(output_dir, accent, f"{accent}_{random_string()}.wav")
    
    # Record audio using FFmpeg
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'dshow',  # DirectShow capture
            '-i', 'audio=Microphone Array (Realtek(R) Audio)',  # Input device
            '-t', str(duration),  # Duration
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian audio codec
            '-ar', '16000',  # 16 kHz sample rate
            '-ac', '1',  # Mono
            output_path
        ]
        
        print("Recording will start in 3 seconds...")
        time.sleep(3)
        print("Recording...")
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the audio file was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully recorded audio to {output_path}")
            return output_path
        else:
            print("Failed to record audio with FFmpeg.")
    
    except Exception as e:
        print(f"Error recording audio: {str(e)}")
    
    print("Failed to record audio.")
    return None

def update_metadata(dataset_dir, accent, audio_path, source, url=None):
    """
    Update the dataset metadata.
    
    Args:
        dataset_dir (str): Dataset directory.
        accent (str): Accent label.
        audio_path (str): Path to the audio file.
        source (str): Source of the audio (youtube, file, recording).
        url (str, optional): URL of the source (for YouTube videos).
    """
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    
    # Load existing metadata if it exists
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"samples": []}
    
    # Add new sample to metadata
    sample = {
        "file_path": os.path.relpath(audio_path, dataset_dir),
        "accent": accent,
        "source": source,
        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if url:
        sample["url"] = url
    
    metadata["samples"].append(sample)
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def main_menu():
    """Display the main menu and handle user input."""
    while True:
        print("\n=== Accent Detection Dataset Collection ===")
        print("1. Download from YouTube")
        print("2. Process existing audio file")
        print("3. Record audio from microphone")
        print("4. View dataset statistics")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            url = input("Enter YouTube URL: ")
            accent = select_accent()
            if accent:
                audio_path = download_from_youtube(url, dataset_dir, accent)
                if audio_path:
                    update_metadata(dataset_dir, accent, audio_path, "youtube", url)
        
        elif choice == '2':
            file_path = input("Enter path to audio file: ")
            accent = select_accent()
            if accent and os.path.exists(file_path):
                audio_path = process_existing_audio(file_path, dataset_dir, accent)
                if audio_path:
                    update_metadata(dataset_dir, accent, audio_path, "file")
            else:
                print("File not found.")
        
        elif choice == '3':
            duration = input("Enter recording duration in seconds (default: 5): ")
            try:
                duration = int(duration) if duration else 5
            except ValueError:
                duration = 5
            
            accent = select_accent()
            if accent:
                audio_path = record_audio(dataset_dir, accent, duration)
                if audio_path:
                    update_metadata(dataset_dir, accent, audio_path, "recording")
        
        elif choice == '4':
            display_statistics()
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

def select_accent():
    """Prompt the user to select an accent."""
    print("\nSelect accent:")
    for i, accent in enumerate(accents):
        print(f"{i+1}. {accent.capitalize()}")
    
    choice = input("\nEnter your choice (1-5): ")
    
    try:
        index = int(choice) - 1
        if 0 <= index < len(accents):
            return accents[index]
        else:
            print("Invalid choice.")
            return None
    except ValueError:
        print("Invalid choice.")
        return None

def display_statistics():
    """Display dataset statistics."""
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print("No metadata found. The dataset is empty.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    samples = metadata["samples"]
    
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(samples)}")
    
    # Count samples by accent
    accent_counts = {}
    for sample in samples:
        accent = sample["accent"]
        accent_counts[accent] = accent_counts.get(accent, 0) + 1
    
    print("\nSamples by accent:")
    for accent in accents:
        count = accent_counts.get(accent, 0)
        print(f"- {accent.capitalize()}: {count}")
    
    # Count samples by source
    source_counts = {}
    for sample in samples:
        source = sample["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\nSamples by source:")
    for source, count in source_counts.items():
        print(f"- {source.capitalize()}: {count}")

if __name__ == "__main__":
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        install_dependencies(missing_deps)
    
    # Create dataset structure
    dataset_dir, accents = create_dataset_structure()
    
    # Display main menu
    main_menu()
