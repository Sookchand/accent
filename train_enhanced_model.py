#!/usr/bin/env python3
"""
Enhanced training script for accent detection with improved model architecture.

This script implements advanced techniques to improve accent detection confidence:
- Enhanced feature extraction
- Advanced model architectures
- Cross-validation and hyperparameter tuning
- Model ensemble techniques
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle
import time
from accent_model import AccentDetector

def load_enhanced_dataset():
    """Load the dataset with enhanced metadata."""
    dataset_dir = "dataset"
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print("❌ Dataset metadata not found. Please run collect_curated_dataset.py first.")
        return None, None, None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    samples = metadata["samples"]
    
    if len(samples) < 10:
        print(f"❌ Dataset has only {len(samples)} samples. Need at least 10 samples for training.")
        return None, None, None
    
    audio_paths = []
    accent_labels = []
    sample_metadata = []
    
    for sample in samples:
        file_path = os.path.join(dataset_dir, sample["file_path"])
        if os.path.exists(file_path):
            audio_paths.append(file_path)
            accent_labels.append(sample["accent"])
            sample_metadata.append(sample)
    
    print(f"✅ Loaded {len(audio_paths)} samples from dataset")
    return audio_paths, accent_labels, sample_metadata

def extract_enhanced_features(detector, audio_paths, accent_labels):
    """Extract enhanced features from audio files."""
    print("Extracting enhanced features from audio files...")
    
    features = []
    valid_indices = []
    
    for i, audio_path in enumerate(audio_paths):
        print(f"Processing {os.path.basename(audio_path)} ({i+1}/{len(audio_paths)})")
        
        # Extract features using the enhanced detector
        feature_vector = detector.extract_features(audio_path)
        
        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(i)
        else:
            print(f"⚠️ Failed to extract features from {audio_path}")
    
    # Filter labels to match valid features
    valid_labels = [accent_labels[i] for i in valid_indices]
    
    print(f"✅ Successfully extracted features from {len(features)} files")
    return np.array(features), valid_labels

def create_advanced_models():
    """Create a collection of advanced models for ensemble."""
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        ),
        'svm': SVC(
            C=10.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
    }
    
    return models

def train_enhanced_model(features, labels, use_ensemble=True, tune_hyperparams=False):
    """Train an enhanced accent detection model."""
    print(f"Training enhanced model with {len(features)} samples...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42,
        stratify=encoded_labels if len(np.unique(encoded_labels)) <= len(encoded_labels) * 0.2 else None
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if use_ensemble:
        # Create ensemble model
        print("Creating ensemble model...")
        models = create_advanced_models()
        
        # Create voting classifier
        estimators = [(name, model) for name, model in models.items()]
        ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
        
        # Train ensemble model
        ensemble_model.fit(X_train_scaled, y_train)
        
        # Apply calibration for better confidence scores
        print("Applying confidence calibration...")
        calibrated_model = CalibratedClassifierCV(ensemble_model, cv=3, method='sigmoid')
        calibrated_model.fit(X_train_scaled, y_train)
        
        final_model = calibrated_model
        
    else:
        # Use single best model
        print("Training single Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        )
        
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        
        model.fit(X_train_scaled, y_train)
        
        # Apply calibration
        print("Applying confidence calibration...")
        final_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
        final_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    print("Evaluating model performance...")
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    except Exception as e:
        print(f"Cross-validation failed: {str(e)}")
    
    # Classification report
    accent_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=accent_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=accent_names, yticklabels=accent_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Enhanced Model')
    
    # Save plots
    os.makedirs("models", exist_ok=True)
    plt.savefig(os.path.join("models", "enhanced_confusion_matrix.png"))
    plt.close()
    
    # Plot confidence distribution
    max_probs = np.max(y_pred_proba, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs * 100, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score (%)')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores - Enhanced Model')
    plt.savefig(os.path.join("models", "enhanced_confidence_distribution.png"))
    plt.close()
    
    return final_model, scaler, label_encoder, accuracy

def save_enhanced_model(model, scaler, label_encoder, accuracy):
    """Save the enhanced model and associated components."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "enhanced_accent_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = os.path.join(models_dir, "enhanced_accent_model_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save label encoder
    encoder_path = os.path.join(models_dir, "enhanced_accent_model_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save model info
    info = {
        "model_type": "enhanced_ensemble",
        "accuracy": accuracy,
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accent_classes": label_encoder.classes_.tolist()
    }
    
    info_path = os.path.join(models_dir, "enhanced_model_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Enhanced model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    print(f"✅ Label encoder saved to {encoder_path}")
    print(f"✅ Model info saved to {info_path}")

def main():
    """Main function."""
    print("=== Enhanced Accent Detection Model Training ===\n")
    
    # Load dataset
    audio_paths, accent_labels, sample_metadata = load_enhanced_dataset()
    if audio_paths is None:
        return
    
    # Initialize detector for feature extraction
    detector = AccentDetector()
    
    # Extract features
    features, valid_labels = extract_enhanced_features(detector, audio_paths, accent_labels)
    
    if len(features) < 5:
        print("❌ Not enough valid samples for training. Need at least 5 samples.")
        return
    
    # Ask for training options
    print("\nTraining Options:")
    print("1. Use ensemble model (recommended)")
    print("2. Use single model")
    
    choice = input("Enter your choice (1-2, default: 1): ")
    use_ensemble = choice != '2'
    
    tune_choice = input("Perform hyperparameter tuning? (y/n, default: n): ")
    tune_hyperparams = tune_choice.lower() == 'y'
    
    # Train model
    start_time = time.time()
    model, scaler, label_encoder, accuracy = train_enhanced_model(
        features, valid_labels, use_ensemble, tune_hyperparams
    )
    training_time = time.time() - start_time
    
    # Save model
    save_enhanced_model(model, scaler, label_encoder, accuracy)
    
    print(f"\n=== Training Complete ===")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final accuracy: {accuracy:.3f}")
    print(f"Model type: {'Ensemble' if use_ensemble else 'Single'}")
    
    print("\n=== Next Steps ===")
    print("1. Update your accent_model.py to use the enhanced model")
    print("2. Test the model with 'streamlit run app.py'")
    print("3. Deploy to Streamlit Cloud for public access")

if __name__ == "__main__":
    main()
