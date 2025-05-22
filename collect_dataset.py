#!/usr/bin/env python3
"""
Script to collect a dataset for accent detection.

This script helps you collect audio samples from various sources to build a dataset
for training an accent detection model. It can download audio from YouTube videos,
process existing audio files, and organize them into a structured dataset.

Usage:
    python collect_dataset.py --output_dir /path/to/dataset --source youtube --urls urls.txt --accent american
    python collect_dataset.py --output_dir /path/to/dataset --source directory --input_dir /path/to/audio --accent british

For detailed guidance on developing a sophisticated accent detection model,
refer to the ADVANCED_MODEL.md file.
"""

import os
import argparse
import glob
import csv
import subprocess
import random
import string
from app import download_video, extract_audio

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Collect a dataset for accent detection.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save the dataset')
    parser.add_argument('--source', type=str, required=True, choices=['youtube', 'directory'],
                        help='Source of audio samples: youtube or directory')
    parser.add_argument('--urls', type=str,
                        help='Path to a text file containing YouTube URLs (one per line)')
    parser.add_argument('--input_dir', type=str,
                        help='Path to a directory containing audio files')
    parser.add_argument('--accent', type=str, required=True,
                        help='Accent label for the collected samples')
    parser.add_argument('--segment_length', type=int, default=5,
                        help='Length of audio segments in seconds')
    parser.add_argument('--max_segments', type=int, default=10,
                        help='Maximum number of segments to extract from each source')
    return parser.parse_args()

def random_string(length=8):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def segment_audio(input_path, output_dir, segment_length=5, max_segments=10):
    """
    Segment an audio file into shorter clips.
    
    Args:
        input_path (str): Path to the input audio file.
        output_dir (str): Directory to save the segmented audio files.
        segment_length (int): Length of each segment in seconds.
        max_segments (int): Maximum number of segments to extract.
        
    Returns:
        list: Paths to the segmented audio files.
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get audio duration using ffprobe
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            duration = float(result.stdout.strip())
        except:
            print(f"Warning: Could not determine duration of {input_path}. Using default of 60 seconds.")
            duration = 60
        
        # Calculate number of segments
        num_segments = min(max_segments, int(duration / segment_length))
        
        if num_segments <= 0:
            print(f"Warning: Audio file {input_path} is too short for segmentation.")
            return []
        
        # Extract segments
        segment_paths = []
        
        for i in range(num_segments):
            # Calculate start time
            start_time = i * segment_length
            
            # Generate output path
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_segment_{i+1}.wav")
            
            # Extract segment using ffmpeg
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(segment_length),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                output_path
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                segment_paths.append(output_path)
            except:
                print(f"Warning: Failed to extract segment {i+1} from {input_path}.")
        
        return segment_paths
        
    except Exception as e:
        print(f"Error segmenting audio: {str(e)}")
        return []

def collect_from_youtube(urls_file, output_dir, accent, segment_length=5, max_segments=10):
    """
    Collect audio samples from YouTube videos.
    
    Args:
        urls_file (str): Path to a text file containing YouTube URLs.
        output_dir (str): Directory to save the dataset.
        accent (str): Accent label for the collected samples.
        segment_length (int): Length of audio segments in seconds.
        max_segments (int): Maximum number of segments to extract from each video.
        
    Returns:
        list: Paths to the collected audio samples.
    """
    # Create output directories
    accent_dir = os.path.join(output_dir, accent)
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(accent_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Read URLs from file
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print(f"No URLs found in {urls_file}")
        return []
    
    print(f"Found {len(urls)} URLs in {urls_file}")
    
    # Process each URL
    all_segments = []
    
    for i, url in enumerate(urls):
        print(f"Processing URL {i+1}/{len(urls)}: {url}")
        
        # Generate temporary file paths
        video_path = os.path.join(temp_dir, f"video_{random_string()}")
        audio_path = os.path.join(temp_dir, f"audio_{random_string()}.wav")
        
        # Download video
        if not download_video(url, video_path):
            print(f"Failed to download video from {url}")
            continue
        
        # Extract audio
        if not extract_audio(video_path, audio_path):
            print(f"Failed to extract audio from {url}")
            continue
        
        # Segment audio
        segments = segment_audio(audio_path, accent_dir, segment_length, max_segments)
        all_segments.extend(segments)
        
        print(f"Extracted {len(segments)} segments from {url}")
        
        # Clean up temporary files
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except:
            pass
    
    # Clean up temporary directory
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    return all_segments

def collect_from_directory(input_dir, output_dir, accent, segment_length=5, max_segments=10):
    """
    Collect audio samples from a directory.
    
    Args:
        input_dir (str): Path to a directory containing audio files.
        output_dir (str): Directory to save the dataset.
        accent (str): Accent label for the collected samples.
        segment_length (int): Length of audio segments in seconds.
        max_segments (int): Maximum number of segments to extract from each file.
        
    Returns:
        list: Paths to the collected audio samples.
    """
    # Create output directory
    accent_dir = os.path.join(output_dir, accent)
    os.makedirs(accent_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
        audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return []
    
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    
    # Process each audio file
    all_segments = []
    
    for i, audio_path in enumerate(audio_files):
        print(f"Processing file {i+1}/{len(audio_files)}: {os.path.basename(audio_path)}")
        
        # Segment audio
        segments = segment_audio(audio_path, accent_dir, segment_length, max_segments)
        all_segments.extend(segments)
        
        print(f"Extracted {len(segments)} segments from {audio_path}")
    
    return all_segments

def create_metadata_file(output_dir, accent, segment_paths):
    """
    Create a metadata CSV file for the collected samples.
    
    Args:
        output_dir (str): Directory to save the dataset.
        accent (str): Accent label for the collected samples.
        segment_paths (list): Paths to the collected audio samples.
        
    Returns:
        str: Path to the metadata file.
    """
    metadata_path = os.path.join(output_dir, f"{accent}_metadata.csv")
    
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'accent'])
        
        for path in segment_paths:
            writer.writerow([os.path.relpath(path, output_dir), accent])
    
    return metadata_path

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect audio samples
    if args.source == 'youtube':
        if not args.urls:
            print("Error: --urls is required for YouTube source")
            return
        
        segment_paths = collect_from_youtube(
            args.urls, args.output_dir, args.accent,
            args.segment_length, args.max_segments
        )
    elif args.source == 'directory':
        if not args.input_dir:
            print("Error: --input_dir is required for directory source")
            return
        
        segment_paths = collect_from_directory(
            args.input_dir, args.output_dir, args.accent,
            args.segment_length, args.max_segments
        )
    
    # Create metadata file
    if segment_paths:
        metadata_path = create_metadata_file(args.output_dir, args.accent, segment_paths)
        print(f"Metadata saved to {metadata_path}")
        print(f"Collected {len(segment_paths)} audio samples for accent '{args.accent}'")
    else:
        print(f"No audio samples collected for accent '{args.accent}'")

if __name__ == "__main__":
    main()
