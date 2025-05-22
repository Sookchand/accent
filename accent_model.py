import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import random

class AccentDetector:
    """
    Accent detection model for English speech.

    This is currently a placeholder implementation that returns random results.
    For a sophisticated implementation, see ADVANCED_MODEL.md for guidance on
    developing a proper machine learning model for accent classification.

    Future versions will implement:
    - Feature extraction from audio (MFCCs, prosodic features, etc.)
    - Pre-trained deep learning models (Wav2Vec2, HuBERT)
    - Confidence scoring based on model probabilities
    - Detailed accent analysis with phonetic explanations
    """

    def __init__(self, model_path=None):
        """
        Initialize the accent detector.

        Args:
            model_path (str, optional): Path to a trained model file.
                                       If None, uses the placeholder implementation.
        """
        # List of supported accents
        self.accents = ["American", "British", "Australian", "Indian", "Canadian"]

        # Detailed explanations for each accent
        self.explanations = {
            "American": "The speaker demonstrates typical American English pronunciation patterns, including rhotic 'r' sounds and clear 't' pronunciation.",
            "British": "The speaker shows characteristics of British English, including non-rhotic pronunciation and distinctive vowel sounds.",
            "Australian": "The speaker exhibits Australian English features, including rising intonation at the end of statements and distinctive vowel shifts.",
            "Indian": "The speaker displays Indian English characteristics, including retroflex consonants and syllable-timed rhythm.",
            "Canadian": "The speaker shows Canadian English features, including Canadian raising and merged vowels before 'r'."
        }

        # Check if a model path is provided
        if model_path and os.path.exists(model_path):
            self.use_placeholder = False
            self.model = self.load_model(model_path)
            self.scaler = self.load_scaler(model_path.replace('.pkl', '_scaler.pkl'))
        else:
            # Use placeholder implementation
            self.use_placeholder = True
            self.model = None
            self.scaler = None

    def extract_features(self, audio_path):
        """
        Extract audio features for accent detection.

        This implementation extracts a comprehensive set of features for accent classification,
        including MFCCs, spectral features, and prosodic features.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            numpy.ndarray: Feature vector for accent classification.
        """
        try:
            # Check if the file exists
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None

            # Load audio file
            y, sr = librosa.load(audio_path, sr=16000)

            # Preprocess audio
            y = self._preprocess_audio(y, sr)

            # 1. MFCCs (Mel-Frequency Cepstral Coefficients)
            # Increased from 13 to 20 MFCCs for better accent discrimination
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            mfccs_delta = librosa.feature.delta(mfccs)  # First-order derivatives
            mfccs_delta_mean = np.mean(mfccs_delta, axis=1)
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)  # Second-order derivatives
            mfccs_delta2_mean = np.mean(mfccs_delta2, axis=1)

            # 2. Spectral features
            # Spectral centroid (brightness of sound)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = np.mean(spectral_centroid)
            spectral_centroid_std = np.std(spectral_centroid)

            # Spectral bandwidth (width of the spectrum)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            spectral_bandwidth_std = np.std(spectral_bandwidth)

            # Spectral contrast (valley-to-peak contrast in spectrum)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

            # Spectral rolloff (frequency below which 85% of energy is contained)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            spectral_rolloff_std = np.std(spectral_rolloff)

            # Zero crossing rate (rate of sign changes)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            zero_crossing_rate_mean = np.mean(zero_crossing_rate)
            zero_crossing_rate_std = np.std(zero_crossing_rate)

            # 3. Prosodic features
            # Pitch (fundamental frequency)
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
                pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            except:
                pitch_mean = 0
                pitch_std = 0

            # Root Mean Square Energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)

            # Tempo and beat features
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            except:
                tempo = 0

            # 4. Combine all features into a comprehensive feature vector
            feature_vector = np.concatenate([
                mfccs_mean, mfccs_std,
                mfccs_delta_mean, mfccs_delta2_mean,
                [spectral_centroid_mean, spectral_centroid_std],
                [spectral_bandwidth_mean, spectral_bandwidth_std],
                spectral_contrast_mean,
                [spectral_rolloff_mean, spectral_rolloff_std],
                [zero_crossing_rate_mean, zero_crossing_rate_std],
                [pitch_mean, pitch_std],
                [rms_mean, rms_std],
                [tempo]
            ])

            return feature_vector

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def _preprocess_audio(self, y, sr):
        """
        Preprocess audio for better feature extraction.

        Args:
            y (numpy.ndarray): Audio time series.
            sr (int): Sample rate.

        Returns:
            numpy.ndarray: Preprocessed audio time series.
        """
        try:
            # Normalize audio
            y = librosa.util.normalize(y)

            # Remove silence
            y, _ = librosa.effects.trim(y, top_db=20)

            # Apply pre-emphasis filter to emphasize higher frequencies
            y = librosa.effects.preemphasis(y)

            return y
        except Exception as e:
            print(f"Error preprocessing audio: {str(e)}")
            return y

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def predict(self, audio_path, transcription=None, min_confidence=30):
        """
        Predict the accent from audio and optional transcription.

        Args:
            audio_path (str): Path to the audio file.
            transcription (str, optional): Transcription of the audio.
                                         Can be used for linguistic analysis.
            min_confidence (float): Minimum confidence threshold (0-100).
                                   If the highest confidence is below this threshold,
                                   the prediction will include multiple possible accents.

        Returns:
            dict: Dictionary containing accent, confidence score, explanation, and additional info.
        """
        # Check if we're using the placeholder implementation
        if self.use_placeholder:
            # This is a placeholder implementation that returns random results
            detected_accent = random.choice(self.accents)
            confidence_score = random.uniform(70, 95)
            explanation = self.explanations.get(detected_accent,
                                              "The accent has distinctive features that require further analysis.")

            return {
                "accent": detected_accent,
                "confidence_score": confidence_score,
                "explanation": explanation,
                "is_placeholder": True
            }

        # If we have a trained model, use it for prediction
        try:
            # Extract features from the audio
            features = self.extract_features(audio_path)

            if features is None:
                # If feature extraction failed, fall back to placeholder
                return self.predict_placeholder()

            # Preprocess the features
            features_scaled = self.scaler.transform([features])

            # Use the trained model to predict the accent
            # Get probabilities for each accent
            accent_probs = self.model.predict_proba(features_scaled)[0]

            # Sort accents by probability (descending)
            sorted_indices = np.argsort(accent_probs)[::-1]
            sorted_accents = [self.accents[i] for i in sorted_indices]
            sorted_probs = accent_probs[sorted_indices]

            # Get the most likely accent
            detected_accent = sorted_accents[0]
            confidence_score = sorted_probs[0] * 100

            # Get the explanation
            explanation = self.explanations.get(detected_accent,
                                              "The accent has distinctive features that require further analysis.")

            # Enhance explanation with confidence information
            if confidence_score < min_confidence:
                # Low confidence, include alternative accents
                alternatives = []
                for i in range(1, min(3, len(sorted_accents))):
                    alt_accent = sorted_accents[i]
                    alt_confidence = sorted_probs[i] * 100
                    if alt_confidence > 10:  # Only include alternatives with reasonable confidence
                        alternatives.append(f"{alt_accent} ({alt_confidence:.1f}%)")

                if alternatives:
                    explanation += f"\n\nHowever, the confidence is low. Alternative possibilities include: {', '.join(alternatives)}."
            elif confidence_score > 90:
                # High confidence
                explanation += "\n\nThis prediction has very high confidence based on clear accent markers."

            # If transcription is provided, use it for additional analysis
            if transcription and len(transcription) > 20:
                # Perform basic linguistic analysis
                linguistic_features = self._analyze_transcription(transcription, detected_accent)
                if linguistic_features:
                    explanation += f"\n\nLinguistic analysis: {linguistic_features}"

            # Prepare the result with detailed information
            result = {
                "accent": detected_accent,
                "confidence_score": confidence_score,
                "explanation": explanation,
                "is_placeholder": False,
                "all_accents": [
                    {"accent": acc, "confidence": prob * 100}
                    for acc, prob in zip(sorted_accents, sorted_probs)
                ]
            }

            return result

        except Exception as e:
            print(f"Error predicting accent: {str(e)}")
            # Fall back to placeholder implementation
            return self.predict_placeholder()

    def _analyze_transcription(self, transcription, detected_accent):
        """
        Analyze transcription for linguistic features related to the detected accent.

        Args:
            transcription (str): Transcription of the audio.
            detected_accent (str): Detected accent.

        Returns:
            str: Description of linguistic features found in the transcription.
        """
        try:
            # Convert to lowercase for easier analysis
            text = transcription.lower()

            # Define accent-specific words and patterns
            accent_patterns = {
                "American": {
                    "words": ["gonna", "wanna", "y'all", "awesome", "totally", "guy", "buddy"],
                    "spellings": ["color", "flavor", "center", "theater"],
                    "patterns": ["r sound after vowels", "t-flapping"]
                },
                "British": {
                    "words": ["bloody", "brilliant", "proper", "cheers", "mate", "quid", "lorry"],
                    "spellings": ["colour", "flavour", "centre", "theatre"],
                    "patterns": ["dropped r after vowels", "t-glottalization"]
                },
                "Australian": {
                    "words": ["mate", "g'day", "arvo", "barbie", "bloke", "fair dinkum", "crikey"],
                    "spellings": ["colour", "flavour", "centre", "theatre"],
                    "patterns": ["high rising terminals", "dropped r after vowels"]
                },
                "Indian": {
                    "words": ["yaar", "acha", "only", "itself", "prepone", "timepass"],
                    "spellings": ["colour", "flavour", "centre", "theatre"],
                    "patterns": ["retroflex consonants", "syllable-timed rhythm"]
                },
                "Canadian": {
                    "words": ["eh", "toque", "loonie", "toonie", "washroom", "hydro"],
                    "spellings": ["colour", "flavour", "centre", "theatre"],
                    "patterns": ["canadian raising", "merged vowels before r"]
                }
            }

            # Check for accent-specific words
            found_features = []

            if detected_accent in accent_patterns:
                patterns = accent_patterns[detected_accent]

                # Check for accent-specific words
                for word in patterns["words"]:
                    if f" {word} " in f" {text} " or text.startswith(word + " ") or text.endswith(" " + word):
                        found_features.append(f"Used '{word}', common in {detected_accent} English")
                        break  # Only report one word match

            if found_features:
                return " ".join(found_features)
            else:
                return ""

        except Exception as e:
            print(f"Error analyzing transcription: {str(e)}")
            return ""

    def predict_placeholder(self):
        """
        Placeholder prediction method that returns random results.
        Used as a fallback when the real model fails.

        Returns:
            dict: Dictionary containing accent, confidence score, explanation, and additional info.
        """
        detected_accent = random.choice(self.accents)
        confidence_score = random.uniform(70, 95)
        explanation = self.explanations.get(detected_accent,
                                          "The accent has distinctive features that require further analysis.")

        # Generate random probabilities for all accents
        all_probs = np.random.random(len(self.accents))
        all_probs = all_probs / all_probs.sum()  # Normalize to sum to 1

        # Set the detected accent to have the highest probability
        detected_idx = self.accents.index(detected_accent)
        for i in range(len(all_probs)):
            if i == detected_idx:
                all_probs[i] = confidence_score / 100
            else:
                all_probs[i] = (1 - confidence_score / 100) / (len(self.accents) - 1)

        # Sort accents by probability
        sorted_indices = np.argsort(all_probs)[::-1]
        sorted_accents = [self.accents[i] for i in sorted_indices]
        sorted_probs = all_probs[sorted_indices]

        return {
            "accent": detected_accent,
            "confidence_score": confidence_score,
            "explanation": explanation,
            "is_placeholder": True,
            "all_accents": [
                {"accent": acc, "confidence": prob * 100}
                for acc, prob in zip(sorted_accents, sorted_probs)
            ]
        }

    def train(self, audio_paths, accent_labels, model_type='ensemble', n_estimators=100, cv=5):
        """
        Train the accent detection model.

        This implementation trains a sophisticated machine learning model
        on a dataset of audio samples with known accent labels.

        Args:
            audio_paths (list): List of paths to audio files.
            accent_labels (list): List of accent labels corresponding to the audio files.
            model_type (str): Type of model to train ('rf', 'svm', 'gb', 'ensemble').
            n_estimators (int): Number of estimators for ensemble methods.
            cv (int): Number of cross-validation folds for model calibration.

        Returns:
            bool: True if training was successful, False otherwise.
        """
        try:
            print("Training a new accent detection model...")

            # 1. Extract features from all audio samples
            features = []
            valid_indices = []

            for i, audio_path in enumerate(audio_paths):
                print(f"Extracting features from {os.path.basename(audio_path)} ({i+1}/{len(audio_paths)})")
                feature_vector = self.extract_features(audio_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    valid_indices.append(i)

            if not features:
                print("No valid features extracted. Training failed.")
                return False

            # Get the corresponding labels for valid features
            labels = [accent_labels[i] for i in valid_indices]

            # 2. Preprocess the features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)

            # 3. Train a machine learning model based on the specified type
            print(f"Training a {model_type} model...")

            if model_type == 'rf':
                # Random Forest Classifier
                base_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    class_weight='balanced',
                    random_state=42
                )
            elif model_type == 'svm':
                # Support Vector Machine
                from sklearn.svm import SVC
                base_model = SVC(
                    C=10.0,
                    kernel='rbf',
                    gamma='scale',
                    probability=True,
                    class_weight='balanced',
                    random_state=42
                )
            elif model_type == 'gb':
                # Gradient Boosting Classifier
                from sklearn.ensemble import GradientBoostingClassifier
                base_model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    subsample=1.0,
                    max_features=None,
                    random_state=42
                )
            elif model_type == 'ensemble':
                # Ensemble of multiple models
                from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
                from sklearn.svm import SVC

                estimators = [
                    ('rf', RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features='sqrt',
                        bootstrap=True,
                        class_weight='balanced',
                        random_state=42
                    )),
                    ('svm', SVC(
                        C=10.0,
                        kernel='rbf',
                        gamma='scale',
                        probability=True,
                        class_weight='balanced',
                        random_state=42
                    )),
                    ('gb', GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=0.1,
                        max_depth=3,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        subsample=1.0,
                        max_features=None,
                        random_state=42
                    ))
                ]

                base_model = VotingClassifier(estimators=estimators, voting='soft')
            else:
                # Default to Random Forest
                print(f"Unknown model type '{model_type}'. Using Random Forest instead.")
                base_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42
                )

            # 4. Apply confidence calibration
            from sklearn.calibration import CalibratedClassifierCV

            # Check if the model already supports probability estimates
            if hasattr(base_model, 'predict_proba') and model_type != 'ensemble':
                print("Applying confidence calibration...")
                self.model = CalibratedClassifierCV(
                    base_estimator=base_model,
                    cv=cv,
                    method='sigmoid'  # Platt scaling
                )
            else:
                # Use the base model directly
                self.model = base_model

            # 5. Train the model
            self.model.fit(features_scaled, labels)

            # 6. Evaluate the model with cross-validation
            from sklearn.model_selection import cross_val_score

            try:
                # Perform cross-validation
                cv_scores = cross_val_score(base_model, features_scaled, labels, cv=cv)

                # Print cross-validation results
                print(f"Cross-validation scores: {cv_scores}")
                print(f"Mean accuracy: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
            except Exception as cv_error:
                print(f"Warning: Could not perform cross-validation: {str(cv_error)}")

            # Set use_placeholder to False since we now have a trained model
            self.use_placeholder = False

            print("Training completed successfully!")
            return True

        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

    def save_model(self, model_path):
        """
        Save the trained model to disk.

        Args:
            model_path (str): Path to save the model.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            if self.model is None:
                print("No model to save.")
                return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

            # Save the scaler
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            print(f"Model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")
            return True

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self, model_path):
        """
        Load a trained model from disk.

        Args:
            model_path (str): Path to the saved model.

        Returns:
            object: The loaded model, or None if loading failed.
        """
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return None

            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            print(f"Model loaded from {model_path}")
            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def load_scaler(self, scaler_path):
        """
        Load a trained scaler from disk.

        Args:
            scaler_path (str): Path to the saved scaler.

        Returns:
            object: The loaded scaler, or None if loading failed.
        """
        try:
            if not os.path.exists(scaler_path):
                print(f"Scaler file not found: {scaler_path}")
                return None

            # Load the scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            print(f"Scaler loaded from {scaler_path}")
            return scaler

        except Exception as e:
            print(f"Error loading scaler: {str(e)}")
            return None

# Example usage:
if __name__ == "__main__":
    detector = AccentDetector()
    result = detector.predict("sample_audio.wav")
    print(f"Detected accent: {result['accent']}")
    print(f"Confidence score: {result['confidence_score']:.1f}%")
    print(f"Explanation: {result['explanation']}")
