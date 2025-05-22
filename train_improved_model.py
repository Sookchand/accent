#!/usr/bin/env python3
"""
Script to train an improved accent detection model.

This script helps you train a more sophisticated accent detection model
using the collected dataset. It provides options for different model types,
hyperparameter tuning, and cross-validation.

Usage:
    python train_improved_model.py

For detailed guidance on developing a sophisticated accent detection model,
refer to the ADVANCED_MODEL.md file.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import pickle
import time
from accent_model import AccentDetector

def check_dataset():
    """Check if the dataset exists and has enough samples."""
    dataset_dir = "dataset"
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    
    if not os.path.exists(dataset_dir):
        print("❌ Dataset directory not found.")
        print("Please run 'python collect_real_dataset.py' to collect data first.")
        return False
    
    if not os.path.exists(metadata_path):
        print("❌ Dataset metadata not found.")
        print("Please run 'python collect_real_dataset.py' to collect data first.")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    samples = metadata["samples"]
    
    if len(samples) < 10:
        print(f"❌ Dataset has only {len(samples)} samples, which is too few for training.")
        print("Please collect at least 10 samples (preferably 50+ samples) using 'python collect_real_dataset.py'.")
        return False
    
    # Count samples by accent
    accent_counts = {}
    for sample in samples:
        accent = sample["accent"]
        accent_counts[accent] = accent_counts.get(accent, 0) + 1
    
    # Check if each accent has at least 2 samples
    min_samples_per_accent = 2
    for accent, count in accent_counts.items():
        if count < min_samples_per_accent:
            print(f"❌ Accent '{accent}' has only {count} samples, which is less than the minimum of {min_samples_per_accent}.")
            print(f"Please collect more samples for the '{accent}' accent using 'python collect_real_dataset.py'.")
            return False
    
    print(f"✅ Dataset has {len(samples)} samples across {len(accent_counts)} accents.")
    return True

def load_dataset():
    """Load the dataset from metadata."""
    dataset_dir = "dataset"
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    samples = metadata["samples"]
    
    audio_paths = []
    accent_labels = []
    
    for sample in samples:
        file_path = os.path.join(dataset_dir, sample["file_path"])
        if os.path.exists(file_path):
            audio_paths.append(file_path)
            accent_labels.append(sample["accent"])
    
    return audio_paths, accent_labels

def train_model(audio_paths, accent_labels, model_type='ensemble', tune_hyperparams=False):
    """
    Train an accent detection model.
    
    Args:
        audio_paths (list): List of paths to audio files.
        accent_labels (list): List of accent labels corresponding to the audio files.
        model_type (str): Type of model to train ('rf', 'svm', 'gb', 'ensemble').
        tune_hyperparams (bool): Whether to perform hyperparameter tuning.
        
    Returns:
        AccentDetector: Trained accent detector.
    """
    print(f"Training a {model_type} model...")
    
    # Initialize the accent detector
    detector = AccentDetector()
    
    # Split the dataset into training and testing sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        audio_paths, accent_labels, test_size=0.2, random_state=42,
        stratify=accent_labels if len(set(accent_labels)) <= len(audio_paths) * 0.2 else None
    )
    
    print(f"Training set size: {len(train_paths)}")
    print(f"Testing set size: {len(test_paths)}")
    
    # Train the model
    if tune_hyperparams:
        print("Performing hyperparameter tuning (this may take a while)...")
        
        # Extract features from all audio samples
        features = []
        valid_indices = []
        
        for i, audio_path in enumerate(train_paths):
            print(f"Extracting features from {os.path.basename(audio_path)} ({i+1}/{len(train_paths)})")
            feature_vector = detector.extract_features(audio_path)
            if feature_vector is not None:
                features.append(feature_vector)
                valid_indices.append(i)
        
        # Get the corresponding labels for valid features
        labels = [train_labels[i] for i in valid_indices]
        
        # Preprocess the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Define hyperparameter grids for different model types
        if model_type == 'rf':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=42)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            }
        elif model_type == 'gb':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:  # ensemble
            # For ensemble, we'll tune each model separately
            print("Hyperparameter tuning for ensemble models is not supported yet.")
            print("Using default hyperparameters for ensemble model.")
            detector.train(train_paths, train_labels, model_type='ensemble')
            return detector, test_paths, test_labels
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(features_scaled, labels)
        
        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
        
        # Train the model with best parameters
        if model_type == 'rf':
            detector.train(train_paths, train_labels, model_type='rf', **grid_search.best_params_)
        elif model_type == 'svm':
            detector.train(train_paths, train_labels, model_type='svm', **grid_search.best_params_)
        elif model_type == 'gb':
            detector.train(train_paths, train_labels, model_type='gb', **grid_search.best_params_)
    else:
        # Train with default parameters
        detector.train(train_paths, train_labels, model_type=model_type)
    
    return detector, test_paths, test_labels

def evaluate_model(detector, test_paths, test_labels):
    """
    Evaluate the trained model.
    
    Args:
        detector (AccentDetector): Trained accent detector.
        test_paths (list): List of paths to test audio files.
        test_labels (list): List of accent labels corresponding to the test files.
        
    Returns:
        float: Accuracy of the model.
    """
    print("Evaluating the model...")
    
    # Make predictions on the test set
    predictions = []
    confidences = []
    
    for i, audio_path in enumerate(test_paths):
        print(f"Testing sample {i+1}/{len(test_paths)}: {os.path.basename(audio_path)}")
        
        # Predict the accent
        result = detector.predict(audio_path)
        predictions.append(result["accent"])
        confidences.append(result["confidence_score"])
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))
    
    # Plot confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(test_labels)),
                yticklabels=sorted(set(test_labels)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot
    os.makedirs("models", exist_ok=True)
    plt.savefig(os.path.join("models", "confusion_matrix.png"))
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, alpha=0.7)
    plt.xlabel('Confidence Score (%)')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    
    # Save the plot
    plt.savefig(os.path.join("models", "confidence_distribution.png"))
    
    return accuracy

def main():
    """Main function."""
    print("=== Improved Accent Detection Model Training ===\n")
    
    # Check if the dataset exists and has enough samples
    if not check_dataset():
        return
    
    # Load the dataset
    audio_paths, accent_labels = load_dataset()
    
    # Ask for model type
    print("\nSelect model type:")
    print("1. Random Forest")
    print("2. Support Vector Machine")
    print("3. Gradient Boosting")
    print("4. Ensemble (combines all of the above)")
    
    model_choice = input("\nEnter your choice (1-4, default: 4): ")
    
    if model_choice == '1':
        model_type = 'rf'
    elif model_choice == '2':
        model_type = 'svm'
    elif model_choice == '3':
        model_type = 'gb'
    else:
        model_type = 'ensemble'
    
    # Ask for hyperparameter tuning
    tune_choice = input("\nPerform hyperparameter tuning? (y/n, default: n): ")
    tune_hyperparams = tune_choice.lower() == 'y'
    
    # Train the model
    start_time = time.time()
    detector, test_paths, test_labels = train_model(
        audio_paths, accent_labels, model_type, tune_hyperparams
    )
    training_time = time.time() - start_time
    
    # Evaluate the model
    accuracy = evaluate_model(detector, test_paths, test_labels)
    
    # Save the model
    model_path = os.path.join("models", "accent_model.pkl")
    detector.save_model(model_path)
    
    print(f"\nModel saved to {model_path}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final accuracy: {accuracy:.2f}")
    
    # Provide instructions for using the model
    print("\n=== Using the Trained Model ===")
    print("The model has been saved and will be automatically loaded by the application.")
    print("Run 'streamlit run app.py' to use the model for accent detection.")

if __name__ == "__main__":
    main()
