#!/usr/bin/env python3
"""
Feature extraction utilities for accent detection.

This module provides functions for extracting audio features for accent detection.
These features can be used to train and evaluate accent detection models.

For detailed guidance on developing a sophisticated accent detection model,
refer to the ADVANCED_MODEL.md file.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def load_audio(audio_path, sr=16000):
    """
    Load an audio file with the specified sample rate.
    
    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sample rate.
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        # Try using librosa
        y, sr = librosa.load(audio_path, sr=sr)
        return y, sr
    except:
        try:
            # Try using soundfile
            y, sr = sf.read(audio_path)
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            # Resample if needed
            if sr != sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr)
            return y, sr
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            return None, None

def extract_mfcc_features(y, sr, n_mfcc=13):
    """
    Extract MFCC features from audio data.
    
    Args:
        y (numpy.ndarray): Audio data.
        sr (int): Sample rate.
        n_mfcc (int): Number of MFCCs to extract.
        
    Returns:
        numpy.ndarray: MFCC features.
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Calculate statistics
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    mfccs_max = np.max(mfccs, axis=1)
    mfccs_min = np.min(mfccs, axis=1)
    
    # Combine features
    features = np.concatenate([mfccs_mean, mfccs_std, mfccs_max, mfccs_min])
    
    return features

def extract_spectral_features(y, sr):
    """
    Extract spectral features from audio data.
    
    Args:
        y (numpy.ndarray): Audio data.
        sr (int): Sample rate.
        
    Returns:
        numpy.ndarray: Spectral features.
    """
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # Combine features
    features = np.array([
        centroid_mean, centroid_std,
        bandwidth_mean, bandwidth_std,
        rolloff_mean, rolloff_std,
        zcr_mean, zcr_std
    ])
    
    # Add contrast features
    features = np.concatenate([features, contrast_mean])
    
    return features

def extract_prosodic_features(y, sr):
    """
    Extract prosodic features from audio data.
    
    Args:
        y (numpy.ndarray): Audio data.
        sr (int): Sample rate.
        
    Returns:
        numpy.ndarray: Prosodic features.
    """
    # Pitch (F0) using PYIN algorithm
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]  # Remove NaN values
        if len(f0) > 0:
            f0_mean = np.mean(f0)
            f0_std = np.std(f0)
            f0_min = np.min(f0)
            f0_max = np.max(f0)
        else:
            f0_mean = f0_std = f0_min = f0_max = 0
    except:
        f0_mean = f0_std = f0_min = f0_max = 0
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # Tempo
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    except:
        tempo = 0
    
    # Combine features
    features = np.array([
        f0_mean, f0_std, f0_min, f0_max,
        rms_mean, rms_std,
        tempo
    ])
    
    return features

def extract_all_features(audio_path, sr=16000):
    """
    Extract all features from an audio file.
    
    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sample rate.
        
    Returns:
        numpy.ndarray: Feature vector.
    """
    # Load audio
    y, sr = load_audio(audio_path, sr)
    if y is None:
        return None
    
    # Extract features
    try:
        # MFCC features
        mfcc_features = extract_mfcc_features(y, sr)
        
        # Spectral features
        spectral_features = extract_spectral_features(y, sr)
        
        # Prosodic features
        prosodic_features = extract_prosodic_features(y, sr)
        
        # Combine all features
        all_features = np.concatenate([mfcc_features, spectral_features, prosodic_features])
        
        return all_features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {str(e)}")
        return None

def extract_features_batch(audio_paths, output_path=None, sr=16000):
    """
    Extract features from a batch of audio files.
    
    Args:
        audio_paths (list): List of paths to audio files.
        output_path (str): Path to save the features as a CSV file.
        sr (int): Target sample rate.
        
    Returns:
        pandas.DataFrame: DataFrame containing the features.
    """
    # Initialize lists for features and file paths
    features_list = []
    valid_paths = []
    
    # Extract features for each audio file
    for audio_path in tqdm(audio_paths, desc="Extracting features"):
        features = extract_all_features(audio_path, sr)
        if features is not None:
            features_list.append(features)
            valid_paths.append(audio_path)
    
    # Create feature names
    n_mfcc = 13
    mfcc_names = []
    for i in range(n_mfcc):
        mfcc_names.extend([f"mfcc{i+1}_mean", f"mfcc{i+1}_std", f"mfcc{i+1}_max", f"mfcc{i+1}_min"])
    
    spectral_names = [
        "centroid_mean", "centroid_std",
        "bandwidth_mean", "bandwidth_std",
        "rolloff_mean", "rolloff_std",
        "zcr_mean", "zcr_std"
    ]
    
    # Add contrast features
    for i in range(7):  # Default number of contrast bands
        spectral_names.append(f"contrast{i+1}_mean")
    
    prosodic_names = [
        "f0_mean", "f0_std", "f0_min", "f0_max",
        "rms_mean", "rms_std",
        "tempo"
    ]
    
    # Combine all feature names
    feature_names = mfcc_names + spectral_names + prosodic_names
    
    # Create DataFrame
    df = pd.DataFrame(features_list, columns=feature_names)
    df["file_path"] = valid_paths
    
    # Save to CSV if output path is provided
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract features from audio files.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing audio files")
    parser.add_argument("--output_path", type=str, default="features.csv",
                        help="Path to save the features as a CSV file")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Target sample rate")
    args = parser.parse_args()
    
    # Get all audio files in the input directory
    audio_paths = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac", ".ogg")):
                audio_paths.append(os.path.join(root, file))
    
    print(f"Found {len(audio_paths)} audio files")
    
    # Extract features
    extract_features_batch(audio_paths, args.output_path, args.sr)
