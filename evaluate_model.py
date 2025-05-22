#!/usr/bin/env python3
"""
Script to evaluate a trained accent detection model.

This script helps you evaluate a trained accent detection model on a test dataset.
It calculates accuracy, precision, recall, and F1-score, and generates a confusion matrix.

Usage:
    python evaluate_model.py --model_path models/accent_model.pkl --test_dir /path/to/test_dataset

For detailed guidance on developing a sophisticated accent detection model,
refer to the ADVANCED_MODEL.md file.
"""

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from accent_model import AccentDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a trained accent detection model.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to the test dataset directory')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                        help='Path to save the evaluation results')
    return parser.parse_args()

def load_test_dataset(test_dir):
    """
    Load the test dataset from the given directory.
    
    Args:
        test_dir (str): Path to the test dataset directory.
        
    Returns:
        tuple: (audio_paths, accent_labels)
    """
    print(f"Loading test dataset from {test_dir}...")
    
    audio_paths = []
    accent_labels = []
    
    # Get all subdirectories (accent categories)
    accent_dirs = [d for d in os.listdir(test_dir) 
                  if os.path.isdir(os.path.join(test_dir, d))]
    
    if not accent_dirs:
        raise ValueError(f"No accent directories found in {test_dir}")
    
    print(f"Found {len(accent_dirs)} accent categories: {', '.join(accent_dirs)}")
    
    # Load audio files for each accent
    for accent in accent_dirs:
        accent_dir = os.path.join(test_dir, accent)
        
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
        raise ValueError("No audio files found in the test dataset")
    
    print(f"Total test dataset size: {len(audio_paths)} audio files")
    
    return audio_paths, accent_labels

def evaluate_model(model_path, audio_paths, accent_labels):
    """
    Evaluate the trained model on the test dataset.
    
    Args:
        model_path (str): Path to the trained model file.
        audio_paths (list): List of paths to audio files.
        accent_labels (list): List of accent labels corresponding to the audio files.
        
    Returns:
        tuple: (accuracy, classification_report_str, confusion_matrix_array, y_true, y_pred)
    """
    print(f"Loading model from {model_path}...")
    
    # Initialize the accent detector with the trained model
    detector = AccentDetector(model_path=model_path)
    
    # Make predictions on the test dataset
    y_true = []
    y_pred = []
    
    for i, (audio_path, true_label) in enumerate(zip(audio_paths, accent_labels)):
        print(f"Testing sample {i+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")
        
        # Predict the accent
        result = detector.predict(audio_path)
        predicted_label = result["accent"]
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=False)
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, report, cm, y_true, y_pred

def plot_confusion_matrix(cm, labels, output_path):
    """
    Plot the confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix.
        labels (list): List of accent labels.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def plot_accuracy_by_accent(y_true, y_pred, output_path):
    """
    Plot the accuracy for each accent.
    
    Args:
        y_true (list): True accent labels.
        y_pred (list): Predicted accent labels.
        output_path (str): Path to save the plot.
    """
    # Calculate accuracy for each accent
    accents = sorted(set(y_true))
    accuracies = []
    
    for accent in accents:
        # Get indices for this accent
        indices = [i for i, label in enumerate(y_true) if label == accent]
        
        # Get true and predicted labels for this accent
        true_labels = [y_true[i] for i in indices]
        pred_labels = [y_pred[i] for i in indices]
        
        # Calculate accuracy
        accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)
        accuracies.append(accuracy * 100)
    
    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    bars = plt.bar(accents, accuracies)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xlabel('Accent')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Accent')
    plt.ylim(0, 105)  # Set y-axis limit to 0-105%
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Accuracy by accent plot saved to {output_path}")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the test dataset
    audio_paths, accent_labels = load_test_dataset(args.test_dir)
    
    # Evaluate the model
    accuracy, report, cm, y_true, y_pred = evaluate_model(
        args.model_path, audio_paths, accent_labels
    )
    
    # Save the classification report
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot the confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, sorted(set(accent_labels)), cm_path)
    
    # Plot accuracy by accent
    acc_path = os.path.join(args.output_dir, 'accuracy_by_accent.png')
    plot_accuracy_by_accent(y_true, y_pred, acc_path)
    
    print("\nEvaluation completed successfully!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
