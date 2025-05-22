#!/usr/bin/env python3
"""
Complete solution implementation for accent detection improvement.

This script automates the entire process:
1. Collects curated dataset with known accents
2. Trains an enhanced model with better architecture
3. Tests the model with your MP4_sample
4. Prepares for Streamlit Cloud deployment
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, title):
    """Print a formatted step."""
    print(f"\n--- Step {step_num}: {title} ---")

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print_step(1, "Checking Dependencies")
    
    required_packages = [
        "streamlit", "google-generativeai", "scikit-learn", 
        "librosa", "matplotlib", "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        return run_command(install_cmd, "Installing missing packages")
    
    return True

def collect_dataset():
    """Collect the curated dataset."""
    print_step(2, "Collecting Curated Dataset")
    
    if not os.path.exists("collect_curated_dataset.py"):
        print("❌ collect_curated_dataset.py not found")
        return False
    
    print("Starting curated dataset collection...")
    try:
        import collect_curated_dataset
        success = collect_curated_dataset.collect_curated_dataset()
        if success:
            print("✅ Dataset collection completed")
            return True
        else:
            print("❌ Dataset collection failed")
            return False
    except Exception as e:
        print(f"❌ Error during dataset collection: {str(e)}")
        return False

def train_enhanced_model():
    """Train the enhanced model."""
    print_step(3, "Training Enhanced Model")
    
    if not os.path.exists("train_enhanced_model.py"):
        print("❌ train_enhanced_model.py not found")
        return False
    
    print("Starting enhanced model training...")
    try:
        import train_enhanced_model
        
        # Load dataset
        audio_paths, accent_labels, sample_metadata = train_enhanced_model.load_enhanced_dataset()
        if audio_paths is None:
            print("❌ Failed to load dataset")
            return False
        
        # Initialize detector for feature extraction
        from accent_model import AccentDetector
        detector = AccentDetector()
        
        # Extract features
        features, valid_labels = train_enhanced_model.extract_enhanced_features(
            detector, audio_paths, accent_labels
        )
        
        if len(features) < 5:
            print("❌ Not enough valid samples for training")
            return False
        
        # Train model with ensemble and calibration
        model, scaler, label_encoder, accuracy = train_enhanced_model.train_enhanced_model(
            features, valid_labels, use_ensemble=True, tune_hyperparams=False
        )
        
        # Save model
        train_enhanced_model.save_enhanced_model(model, scaler, label_encoder, accuracy)
        
        print(f"✅ Enhanced model trained with accuracy: {accuracy:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        return False

def test_with_sample():
    """Test the model with MP4_sample if available."""
    print_step(4, "Testing with Sample File")
    
    sample_path = "MP4_sample"
    if not os.path.exists(sample_path):
        print(f"⚠️ Sample file {sample_path} not found. Skipping test.")
        return True
    
    try:
        from accent_model import AccentDetector
        
        # Initialize detector (should load enhanced model)
        detector = AccentDetector()
        
        # Test prediction
        print(f"Testing with {sample_path}...")
        result = detector.predict(sample_path, "Test transcription")
        
        print(f"✅ Test completed!")
        print(f"   Detected accent: {result['accent']}")
        print(f"   Confidence: {result['confidence_score']:.1f}%")
        print(f"   Using enhanced model: {not result.get('is_placeholder', True)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def prepare_deployment():
    """Prepare files for Streamlit Cloud deployment."""
    print_step(5, "Preparing for Deployment")
    
    # Check required files
    required_files = [
        "app.py", "requirements.txt", "packages.txt", 
        ".streamlit/config.toml", "GEMINI_SETUP.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing deployment files: {', '.join(missing_files)}")
        return False
    
    print("✅ All deployment files are ready")
    
    # Check if .env has API keys
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            env_content = f.read()
            
        if "your_gemini_api_key_here" in env_content:
            print("⚠️ Please update your GEMINI_API_KEY in .env file")
        else:
            print("✅ API keys appear to be configured")
    
    return True

def display_next_steps():
    """Display next steps for the user."""
    print_header("🎉 SOLUTION IMPLEMENTATION COMPLETE!")
    
    print("\n📊 RESULTS SUMMARY:")
    
    # Check if enhanced model exists
    if os.path.exists("models/enhanced_accent_model.pkl"):
        print("✅ Enhanced accent detection model trained and ready")
        
        # Load model info if available
        info_path = "models/enhanced_model_info.json"
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r') as f:
                info = json.load(f)
            print(f"   - Model accuracy: {info.get('accuracy', 'Unknown'):.3f}")
            print(f"   - Accent classes: {', '.join(info.get('accent_classes', []))}")
    else:
        print("⚠️ Enhanced model not found - using placeholder")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Get your Google Gemini API key:")
    print("   - Visit: https://makersuite.google.com/app/apikey")
    print("   - Update your .env file with the key")
    
    print("\n2. Test locally:")
    print("   streamlit run app.py")
    
    print("\n3. Deploy to Streamlit Cloud:")
    print("   - Push code to GitHub")
    print("   - Go to share.streamlit.io")
    print("   - Connect your repository")
    print("   - Add GEMINI_API_KEY to secrets")
    print("   - Deploy!")
    
    print("\n📈 EXPECTED IMPROVEMENTS:")
    print("- Confidence scores: 70-90% (vs previous 40%)")
    print("- Better accent classification accuracy")
    print("- Real transcriptions with Google Gemini API")
    print("- Multiple input methods (URL, upload, local)")
    
    print("\n📚 DOCUMENTATION:")
    print("- README.md: Updated with new features")
    print("- GEMINI_SETUP.md: Google API setup guide")
    print("- DEPLOYMENT.md: Deployment instructions")

def main():
    """Main function to implement the complete solution."""
    print_header("🎯 ACCENT DETECTION SOLUTION IMPLEMENTATION")
    
    print("This script will implement a complete solution to improve your accent detection system:")
    print("1. Install missing dependencies")
    print("2. Collect curated dataset with known accents")
    print("3. Train enhanced model with better architecture")
    print("4. Test with your sample file")
    print("5. Prepare for Streamlit Cloud deployment")
    
    proceed = input("\nDo you want to proceed? (y/n): ")
    if proceed.lower() != 'y':
        print("Implementation cancelled.")
        return
    
    start_time = time.time()
    
    # Execute all steps
    steps = [
        ("Dependencies", check_dependencies),
        ("Dataset Collection", collect_dataset),
        ("Model Training", train_enhanced_model),
        ("Sample Testing", test_with_sample),
        ("Deployment Prep", prepare_deployment)
    ]
    
    completed_steps = 0
    for step_name, step_function in steps:
        if step_function():
            completed_steps += 1
        else:
            print(f"\n❌ Failed at step: {step_name}")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Total execution time: {total_time:.2f} seconds")
    print(f"📊 Completed {completed_steps}/{len(steps)} steps")
    
    if completed_steps == len(steps):
        display_next_steps()
    else:
        print("\n⚠️ Implementation incomplete. Please check the errors above and try again.")

if __name__ == "__main__":
    main()
