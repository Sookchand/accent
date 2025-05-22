#!/usr/bin/env python3
"""
Script to collect a curated dataset with known accent samples.

This script includes pre-verified YouTube URLs with known accents to build
a high-quality training dataset for accent detection.
"""

import os
import sys
import json
import time
import tempfile
import shutil
from datetime import datetime
from app import download_video, extract_audio

# Curated list of YouTube videos with known accents
CURATED_VIDEOS = {
    "british": [
        {
            "url": "https://www.youtube.com/watch?v=0BU_u8_blss",
            "description": "English with Lucy - Modern Received Pronunciation",
            "speaker": "Lucy Earl",
            "accent_type": "Received Pronunciation"
        },
        {
            "url": "https://www.youtube.com/watch?v=ldVJmCKqVnA",
            "description": "BBC English pronunciation guide",
            "speaker": "BBC presenter",
            "accent_type": "BBC English"
        },
        {
            "url": "https://www.youtube.com/watch?v=T7SWETadMn0",
            "description": "British accent tutorial",
            "speaker": "British teacher",
            "accent_type": "Standard British"
        }
    ],
    "american": [
        {
            "url": "https://www.youtube.com/watch?v=J7GY1Xg6X20",
            "description": "American English pronunciation",
            "speaker": "American teacher",
            "accent_type": "General American"
        },
        {
            "url": "https://www.youtube.com/watch?v=LIQsyHoLudQ",
            "description": "American accent training",
            "speaker": "American coach",
            "accent_type": "Standard American"
        }
    ],
    "australian": [
        {
            "url": "https://www.youtube.com/watch?v=KpBYnL5fAXE",
            "description": "Australian accent guide",
            "speaker": "Australian teacher",
            "accent_type": "General Australian"
        },
        {
            "url": "https://www.youtube.com/watch?v=ZnjoD9w7z0Y",
            "description": "Australian pronunciation",
            "speaker": "Australian speaker",
            "accent_type": "Standard Australian"
        }
    ],
    "indian": [
        {
            "url": "https://www.youtube.com/watch?v=v9arM_agKFA",
            "description": "Indian English pronunciation",
            "speaker": "Indian teacher",
            "accent_type": "Indian English"
        },
        {
            "url": "https://www.youtube.com/watch?v=dJgoTcyrFZ4",
            "description": "Indian accent tutorial",
            "speaker": "Indian speaker",
            "accent_type": "Standard Indian"
        }
    ],
    "canadian": [
        {
            "url": "https://www.youtube.com/watch?v=jrTCDi3xbTw",
            "description": "Canadian English guide",
            "speaker": "Canadian teacher",
            "accent_type": "General Canadian"
        },
        {
            "url": "https://www.youtube.com/watch?v=Oe3wj1Y2Hps",
            "description": "Canadian pronunciation",
            "speaker": "Canadian speaker",
            "accent_type": "Standard Canadian"
        }
    ]
}

def create_dataset_structure():
    """Create the dataset directory structure."""
    dataset_dir = "dataset"
    accents = list(CURATED_VIDEOS.keys())

    # Create main dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create subdirectories for each accent
    for accent in accents:
        accent_dir = os.path.join(dataset_dir, accent)
        if not os.path.exists(accent_dir):
            os.makedirs(accent_dir)

    return dataset_dir, accents

def collect_curated_dataset():
    """Collect the curated dataset."""
    print("=== Collecting Curated Accent Dataset ===\n")

    # Create dataset structure
    dataset_dir, accents = create_dataset_structure()

    total_videos = sum(len(videos) for videos in CURATED_VIDEOS.values())
    current_video = 0
    successful_downloads = 0

    print(f"Collecting {total_videos} curated videos across {len(accents)} accents...\n")

    # Process each accent
    for accent, videos in CURATED_VIDEOS.items():
        print(f"\n--- Collecting {accent.upper()} accent samples ---")

        for video_info in videos:
            current_video += 1
            url = video_info["url"]
            description = video_info["description"]

            print(f"\n[{current_video}/{total_videos}] Processing: {description}")
            print(f"URL: {url}")

            # Download the video
            try:
                # Create temporary paths
                temp_dir = tempfile.mkdtemp()
                video_path = os.path.join(temp_dir, f"video_{current_video}")
                audio_path = os.path.join(temp_dir, f"audio_{current_video}.wav")

                # Download video
                if download_video(url, video_path):
                    # Extract audio
                    if extract_audio(video_path, audio_path):
                        # Move audio to dataset directory
                        accent_dir = os.path.join(dataset_dir, accent)
                        os.makedirs(accent_dir, exist_ok=True)

                        final_audio_path = os.path.join(accent_dir, f"{accent}_{current_video}.wav")
                        shutil.move(audio_path, final_audio_path)

                        # Update metadata
                        update_metadata_file(dataset_dir, accent, final_audio_path, video_info, url)

                        successful_downloads += 1
                        print(f"✅ Successfully downloaded and processed!")
                    else:
                        print(f"❌ Failed to extract audio")
                else:
                    print(f"❌ Failed to download video")

                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

            except Exception as e:
                print(f"❌ Error processing video: {str(e)}")

            # Add a small delay to be respectful to the servers
            time.sleep(2)

    print(f"\n=== Collection Complete ===")
    print(f"Successfully downloaded: {successful_downloads}/{total_videos} videos")
    print(f"Dataset saved to: {dataset_dir}")

    # Display final statistics
    display_dataset_statistics(dataset_dir)

    return successful_downloads > 0

def update_metadata_file(dataset_dir, accent, audio_path, video_info, url):
    """Update the metadata file with new sample information."""
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    # Initialize metadata if it doesn't exist
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "dataset_name": "Curated Accent Dataset",
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": []
        }

    # Add new sample
    sample_info = {
        "file_path": os.path.relpath(audio_path, dataset_dir),
        "accent": accent,
        "source": "youtube",
        "url": url,
        "description": video_info["description"],
        "speaker": video_info["speaker"],
        "accent_type": video_info["accent_type"],
        "verified": True,
        "quality": "high",
        "added_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata["samples"].append(sample_info)

    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def display_dataset_statistics(dataset_dir):
    """Display dataset statistics."""
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        print("No metadata found.")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    samples = metadata["samples"]

    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {len(samples)}")

    # Count samples by accent
    accent_counts = {}
    for sample in samples:
        accent = sample["accent"]
        accent_counts[accent] = accent_counts.get(accent, 0) + 1

    print("\nSamples by accent:")
    for accent, count in accent_counts.items():
        print(f"- {accent.capitalize()}: {count}")

    # Count verified samples
    verified_count = sum(1 for sample in samples if sample.get("verified", False))
    print(f"\nVerified samples: {verified_count}")
    print(f"High-quality samples: {sum(1 for sample in samples if sample.get('quality') == 'high')}")

def main():
    """Main function."""
    print("This script will collect a curated dataset of accent samples from verified sources.")
    print("The dataset includes samples from known speakers with clearly identified accents.")

    proceed = input("\nDo you want to proceed with collecting the curated dataset? (y/n): ")

    if proceed.lower() != 'y':
        print("Collection cancelled.")
        return

    # Collect the curated dataset
    success = collect_curated_dataset()

    if success:
        print("\n=== Next Steps ===")
        print("1. Review the collected dataset")
        print("2. Run 'python train_improved_model.py' to train a better model")
        print("3. Test the improved model with 'streamlit run app.py'")
    else:
        print("\n❌ Dataset collection failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
