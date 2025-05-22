#!/usr/bin/env python3
"""
Script to train a new accent detection model.

This script helps you train a new accent detection model using a dataset of audio samples.
For detailed guidance on developing a sophisticated accent detection model,
refer to the ADVANCED_MODEL.md file.

Usage:
    python train_model.py --data_dir /path/to/dataset --output_dir /path/to/output

The dataset directory should have subdirectories for each accent, e.g.:
    /path/to/dataset/
        american/
            sample1.wav
            sample2.wav
            ...
        british/
            sample1.wav
            sample2.wav
            ...
        ...
"""

import os
import argparse
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from accent_model import AccentDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a new accent detection model.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Path to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    return parser.parse_args()

def load_dataset(data_dir):
    """
    Load the dataset from the given directory.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        tuple: (audio_paths, accent_labels)
    """
    print(f"Loading dataset from {data_dir}...")

    audio_paths = []
    accent_labels = []

    # Get all subdirectories (accent categories)
    accent_dirs = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d))]

    if not accent_dirs:
        raise ValueError(f"No accent directories found in {data_dir}")

    print(f"Found {len(accent_dirs)} accent categories: {', '.join(accent_dirs)}")

    # Load audio files for each accent
    for accent in accent_dirs:
        accent_dir = os.path.join(data_dir, accent)

        # Get all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(glob.glob(os.path.join(accent_dir, ext)))

        if not audio_files:
            print(f"Warning: No audio files found for accent '{accent}'")
            continue

        print(f"Found {len(audio_files)} audio files for accent '{accent}'")

        # Add to dataset
        audio_paths.extend(audio_files)
        accent_labels.extend([accent] * len(audio_files))

    if not audio_paths:
        raise ValueError("No audio files found in the dataset")

    print(f"Total dataset size: {len(audio_paths)} audio files")

    return audio_paths, accent_labels

def train_model(audio_paths, accent_labels, test_size=0.2, random_state=42):
    """
    Train a new accent detection model.

    Args:
        audio_paths (list): List of paths to audio files.
        accent_labels (list): List of accent labels corresponding to the audio files.
        test_size (float): Proportion of the dataset to use for testing.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (detector, test_paths, test_labels, accuracy)
    """
    print("Splitting dataset into training and testing sets...")

    # Check if the dataset is too small for stratified split
    unique_labels = set(accent_labels)
    if len(unique_labels) > len(audio_paths) * test_size:
        print(f"Warning: Dataset too small for stratified split with test_size={test_size}")
        print(f"Using simple random split instead (not stratified)")
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            audio_paths, accent_labels, test_size=test_size, random_state=random_state
        )
    else:
        # Split the dataset with stratification
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            audio_paths, accent_labels, test_size=test_size, random_state=random_state,
            stratify=accent_labels
        )

    print(f"Training set size: {len(train_paths)}")
    print(f"Testing set size: {len(test_paths)}")

    # Initialize the accent detector
    detector = AccentDetector()

    # Train the model
    print("Training the model...")
    success = detector.train(train_paths, train_labels)

    if not success:
        raise RuntimeError("Model training failed")

    # Evaluate the model
    print("Evaluating the model...")

    # Make predictions on the test set
    y_true = []
    y_pred = []

    for i, (audio_path, true_label) in enumerate(zip(test_paths, test_labels)):
        print(f"Testing sample {i+1}/{len(test_paths)}: {os.path.basename(audio_path)}")

        # Predict the accent
        result = detector.predict(audio_path)
        predicted_label = result["accent"]

        y_true.append(true_label)
        y_pred.append(predicted_label)

    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return detector, test_paths, test_labels, accuracy

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """
    Plot the confusion matrix.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        output_dir (str): Directory to save the plot.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_true)),
                yticklabels=sorted(set(y_true)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset
    audio_paths, accent_labels = load_dataset(args.data_dir)

    # Train the model
    detector, test_paths, test_labels, accuracy = train_model(
        audio_paths, accent_labels, args.test_size, args.random_state
    )

    # Save the model
    model_path = os.path.join(args.output_dir, 'accent_model.pkl')
    detector.save_model(model_path)

    # Plot the confusion matrix
    y_true = test_labels
    y_pred = [detector.predict(path)["accent"] for path in test_paths]
    plot_confusion_matrix(y_true, y_pred, args.output_dir)

    print("\nTraining completed successfully!")
    print(f"Model saved to {model_path}")
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
