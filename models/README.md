# Accent Detection Models

This directory is used to store trained accent detection models.

## Model Files

When you train a model using the `train_model.py` script, it will save the following files in this directory:

- `accent_model.pkl`: The trained model file
- `accent_model_scaler.pkl`: The feature scaler used for preprocessing

## Training a Model

To train a new accent detection model, use the `train_model.py` script:

```bash
python train_model.py --data_dir /path/to/dataset --output_dir models
```

For more information on training a model, see the [ADVANCED_MODEL.md](../ADVANCED_MODEL.md) file.

## Collecting a Dataset

To collect a dataset for training, use the `collect_dataset.py` script:

```bash
# Collect from YouTube videos
python collect_dataset.py --output_dir dataset --source youtube --urls urls.txt --accent american

# Collect from existing audio files
python collect_dataset.py --output_dir dataset --source directory --input_dir /path/to/audio --accent british
```

## Model Format

The accent detection model is a scikit-learn model that takes audio features as input and outputs accent predictions. The model is trained on a dataset of audio samples with known accent labels.

## Using a Custom Model

If you want to use a custom model, place it in this directory with the name `accent_model.pkl` and the application will automatically load it. Alternatively, you can specify a custom model path when initializing the `AccentDetector` class:

```python
from accent_model import AccentDetector

detector = AccentDetector(model_path="/path/to/your/model.pkl")
```
