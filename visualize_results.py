#!/usr/bin/env python3
"""
Script to visualize accent detection results.

This script helps you visualize the results of accent detection on a dataset.
It generates various plots and visualizations to help understand the model's performance.

Usage:
    python visualize_results.py --model_path models/accent_model.pkl --test_dir dataset/test --output_dir visualizations

For detailed guidance on developing a sophisticated accent detection model,
refer to the ADVANCED_MODEL.md file.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from accent_model import AccentDetector
from feature_extraction import extract_all_features

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize accent detection results.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to the test dataset directory')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Path to save the visualizations')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to use (None for all)')
    return parser.parse_args()

def load_test_dataset(test_dir, num_samples=None):
    """
    Load the test dataset from the given directory.
    
    Args:
        test_dir (str): Path to the test dataset directory.
        num_samples (int): Number of samples to use (None for all).
        
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
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            audio_files.extend([os.path.join(accent_dir, f) for f in os.listdir(accent_dir) if f.endswith(ext)])
        
        if not audio_files:
            print(f"Warning: No audio files found for accent '{accent}'")
            continue
        
        # Limit the number of samples if specified
        if num_samples is not None:
            samples_per_accent = num_samples // len(accent_dirs)
            if samples_per_accent < 1:
                samples_per_accent = 1
            audio_files = audio_files[:samples_per_accent]
        
        print(f"Using {len(audio_files)} audio files for accent '{accent}'")
        
        # Add to dataset
        audio_paths.extend(audio_files)
        accent_labels.extend([accent] * len(audio_files))
    
    if not audio_paths:
        raise ValueError("No audio files found in the test dataset")
    
    print(f"Total test dataset size: {len(audio_paths)} audio files")
    
    return audio_paths, accent_labels

def predict_accents(model_path, audio_paths):
    """
    Predict accents for the given audio files.
    
    Args:
        model_path (str): Path to the trained model file.
        audio_paths (list): List of paths to audio files.
        
    Returns:
        list: List of predicted accents.
    """
    print(f"Loading model from {model_path}...")
    
    # Initialize the accent detector with the trained model
    detector = AccentDetector(model_path=model_path)
    
    # Make predictions
    predictions = []
    confidences = []
    
    for i, audio_path in enumerate(audio_paths):
        print(f"Predicting accent for sample {i+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")
        
        # Predict the accent
        result = detector.predict(audio_path)
        predictions.append(result["accent"])
        confidences.append(result["confidence_score"])
    
    return predictions, confidences

def extract_features_for_visualization(audio_paths):
    """
    Extract features for visualization.
    
    Args:
        audio_paths (list): List of paths to audio files.
        
    Returns:
        numpy.ndarray: Feature matrix.
    """
    print("Extracting features for visualization...")
    
    features = []
    valid_indices = []
    
    for i, audio_path in enumerate(audio_paths):
        print(f"Extracting features for sample {i+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")
        
        # Extract features
        feature_vector = extract_all_features(audio_path)
        
        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(i)
    
    return np.array(features), valid_indices

def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Plot the confusion matrix.
    
    Args:
        y_true (list): True accent labels.
        y_pred (list): Predicted accent labels.
        output_path (str): Path to save the plot.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=sorted(set(y_true)),
                yticklabels=sorted(set(y_true)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def plot_confidence_distribution(confidences, predictions, true_labels, output_path):
    """
    Plot the distribution of confidence scores.
    
    Args:
        confidences (list): Confidence scores.
        predictions (list): Predicted accent labels.
        true_labels (list): True accent labels.
        output_path (str): Path to save the plot.
    """
    # Create a DataFrame
    df = pd.DataFrame({
        'Confidence': confidences,
        'Predicted': predictions,
        'True': true_labels,
        'Correct': [p == t for p, t in zip(predictions, true_labels)]
    })
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Confidence', hue='Correct', bins=20, alpha=0.7)
    plt.xlabel('Confidence Score (%)')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Confidence distribution saved to {output_path}")

def plot_tsne_visualization(features, labels, output_path):
    """
    Plot t-SNE visualization of the features.
    
    Args:
        features (numpy.ndarray): Feature matrix.
        labels (list): Accent labels.
        output_path (str): Path to save the plot.
    """
    # Apply t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'Accent': labels
    })
    
    # Plot t-SNE visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='Accent', palette='viridis', alpha=0.8)
    plt.title('t-SNE Visualization of Accent Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Save the plot
    plt.savefig(output_path)
    print(f"t-SNE visualization saved to {output_path}")

def plot_feature_importance(model_path, feature_names, output_path):
    """
    Plot feature importance.
    
    Args:
        model_path (str): Path to the trained model file.
        feature_names (list): List of feature names.
        output_path (str): Path to save the plot.
    """
    # Load the model
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Check if the model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path)
        print(f"Feature importance plot saved to {output_path}")
    else:
        print("Model does not have feature_importances_ attribute. Skipping feature importance plot.")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the test dataset
    audio_paths, true_labels = load_test_dataset(args.test_dir, args.num_samples)
    
    # Predict accents
    predictions, confidences = predict_accents(args.model_path, audio_paths)
    
    # Extract features for visualization
    features, valid_indices = extract_features_for_visualization(audio_paths)
    
    # Filter true labels and predictions based on valid indices
    true_labels_valid = [true_labels[i] for i in valid_indices]
    predictions_valid = [predictions[i] for i in valid_indices]
    confidences_valid = [confidences[i] for i in valid_indices]
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(true_labels_valid, predictions_valid, cm_path)
    
    # Plot confidence distribution
    conf_path = os.path.join(args.output_dir, 'confidence_distribution.png')
    plot_confidence_distribution(confidences_valid, predictions_valid, true_labels_valid, conf_path)
    
    # Plot t-SNE visualization
    tsne_path = os.path.join(args.output_dir, 'tsne_visualization.png')
    plot_tsne_visualization(features, true_labels_valid, tsne_path)
    
    # Generate feature names
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
    
    # Plot feature importance
    fi_path = os.path.join(args.output_dir, 'feature_importance.png')
    try:
        plot_feature_importance(args.model_path, feature_names, fi_path)
    except:
        print("Failed to plot feature importance. Skipping.")
    
    # Save classification report
    report = classification_report(true_labels_valid, predictions_valid, output_dict=False)
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    print("\nVisualization completed successfully!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
