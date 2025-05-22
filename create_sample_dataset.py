#!/usr/bin/env python3
"""
Script to create a sample dataset for accent detection.

This script creates a small sample dataset for testing the accent detection model.
It generates synthetic audio samples for different accents using text-to-speech.

Usage:
    python create_sample_dataset.py --output_dir dataset/sample

For detailed guidance on developing a sophisticated accent detection model,
refer to the ADVANCED_MODEL.md file.
"""

import os
import argparse
import numpy as np
from scipy.io import wavfile
import random
import string

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create a sample dataset for accent detection.')
    parser.add_argument('--output_dir', type=str, default='dataset/sample',
                        help='Path to save the sample dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples per accent')
    return parser.parse_args()

def random_string(length=8):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def create_sine_wave(freq, duration, sample_rate=16000):
    """
    Create a sine wave.
    
    Args:
        freq (float): Frequency of the sine wave in Hz.
        duration (float): Duration of the sine wave in seconds.
        sample_rate (int): Sample rate in Hz.
        
    Returns:
        numpy.ndarray: The sine wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t) * 32767
    return wave.astype(np.int16)

def create_sample_audio(output_path, accent):
    """
    Create a sample audio file for a given accent.
    
    Args:
        output_path (str): Path to save the audio file.
        accent (str): Accent label.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Create a unique audio sample for each accent
        # In a real implementation, you would use a text-to-speech service
        # with different accent options, or record real speakers
        
        # For now, we'll create synthetic audio with different characteristics
        sample_rate = 16000
        duration = 3.0  # seconds
        
        # Use different frequency ranges for different accents
        if accent == "american":
            freq = random.uniform(200, 300)
        elif accent == "british":
            freq = random.uniform(300, 400)
        elif accent == "australian":
            freq = random.uniform(400, 500)
        elif accent == "indian":
            freq = random.uniform(500, 600)
        elif accent == "canadian":
            freq = random.uniform(600, 700)
        else:
            freq = random.uniform(200, 700)
        
        # Create the sine wave
        wave = create_sine_wave(freq, duration, sample_rate)
        
        # Add some noise
        noise = np.random.normal(0, 500, wave.shape).astype(np.int16)
        wave = np.clip(wave + noise, -32768, 32767).astype(np.int16)
        
        # Save the audio file
        wavfile.write(output_path, sample_rate, wave)
        
        return True
        
    except Exception as e:
        print(f"Error creating sample audio: {str(e)}")
        return False

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Define accents
    accents = ["american", "british", "australian", "indian", "canadian"]
    
    # Create output directories
    for accent in accents:
        accent_dir = os.path.join(args.output_dir, accent)
        os.makedirs(accent_dir, exist_ok=True)
    
    # Create sample audio files
    for accent in accents:
        accent_dir = os.path.join(args.output_dir, accent)
        
        print(f"Creating {args.num_samples} samples for accent '{accent}'...")
        
        for i in range(args.num_samples):
            # Generate output path
            output_path = os.path.join(accent_dir, f"{accent}_{i+1}.wav")
            
            # Create sample audio
            if create_sample_audio(output_path, accent):
                print(f"  Created {output_path}")
            else:
                print(f"  Failed to create {output_path}")
    
    print("\nSample dataset created successfully!")
    print(f"Dataset saved to {args.output_dir}")
    print(f"Total samples: {len(accents) * args.num_samples}")

if __name__ == "__main__":
    main()
