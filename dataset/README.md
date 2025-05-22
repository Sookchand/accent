# Accent Detection Dataset

This directory contains the dataset for training and evaluating accent detection models.

## Directory Structure

The dataset is organized by accent:

- `american/`: American English accent samples
- `british/`: British English accent samples
- `australian/`: Australian English accent samples
- `indian/`: Indian English accent samples
- `canadian/`: Canadian English accent samples

## Collecting Data

You can collect data for this dataset using the `collect_dataset.py` script:

```bash
# Collect from YouTube videos
python collect_dataset.py --output_dir dataset --source youtube --urls urls.txt --accent american

# Collect from existing audio files
python collect_dataset.py --output_dir dataset --source directory --input_dir /path/to/audio --accent british
```

## Creating a Sample Dataset

You can create a synthetic sample dataset using the `create_sample_dataset.py` script:

```bash
python create_sample_dataset.py --output_dir dataset/sample --num_samples 5
```

## Dataset Format

Each audio file should be in WAV, MP3, FLAC, or OGG format. The recommended format is WAV with a sample rate of 16kHz, mono channel, and 16-bit PCM encoding.

## Recommended Dataset Size

For best results, aim for at least:
- 100 samples per accent
- 5-10 seconds of clear speech per sample
- Diverse speakers (age, gender, regional variations)

## Data Sources

Good sources for accent samples include:

1. **Mozilla Common Voice**: https://commonvoice.mozilla.org/
2. **Speech Accent Archive**: https://accent.gmu.edu/
3. **VoxForge**: http://www.voxforge.org/
4. **YouTube videos** with speakers of known accents
5. **Podcasts** with speakers of known accents

## Data Preprocessing

Before training a model, it's recommended to preprocess the audio files:

1. Convert to a consistent format (WAV, 16kHz, mono)
2. Normalize the volume
3. Remove silence and background noise
4. Segment into 5-10 second clips

You can use the `feature_extraction.py` script to extract features from the audio files:

```bash
python feature_extraction.py --input_dir dataset --output_path features.csv
```

## Training and Evaluation

Once you have collected and preprocessed your dataset, you can train and evaluate a model:

```bash
# Train a model
python train_model.py --data_dir dataset --output_dir models

# Evaluate the model
python evaluate_model.py --model_path models/accent_model.pkl --test_dir dataset/test

# Visualize the results
python visualize_results.py --model_path models/accent_model.pkl --test_dir dataset/test --output_dir visualizations
```
