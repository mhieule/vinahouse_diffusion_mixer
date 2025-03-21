# Vinahouse Remix Model

A Python package for transforming songs into Vinahouse style using a hybrid approach combining diffusion models with dedicated components for tempo adjustment and rhythm transformation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Examples](#examples)
- [License](#license)

## Overview

Vinahouse is a style of Vietnamese electronic dance music characterized by fast beats (132-142 BPM), loud bass, and simple rhythms. This project provides a machine learning-based solution for remixing any song into the Vinahouse genre.

The system uses a hybrid approach that combines:
1. **Diffusion models** for spectral transformation
2. **Dedicated tempo adjustment** to match Vinahouse BPM range
3. **Rhythm transformation** to create characteristic beat patterns
4. **Bass enhancement** to achieve the powerful low-end typical of Vinahouse

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended for training)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/vinahouse-remix-project.git
cd vinahouse-remix-project

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Transform a song to Vinahouse style

```bash
# Using the command-line interface
python -m vinahouse_remixer.inference.inference \
    --input path/to/your/song.wav \
    --output path/to/output/directory \
    --model path/to/pretrained/model.pt \
    --config vinahouse_remixer/configs/training_config.json \
    --style-strength 0.8
```

### Using the Python API

```python
from vinahouse_remixer.remixer import VinahouseRemixer

# Initialize the remixer
remixer = VinahouseRemixer(
    model_path="path/to/pretrained/model.pt",
    device="cuda",  # or "cpu"
    style_strength=0.8
)

# Remix a song
output_file = remixer.remix(
    input_file="path/to/your/song.wav",
    output_file="path/to/output/song.wav"
)

print(f"Remixed song saved to: {output_file}")
```

## Usage

### Training

To train the diffusion model on your own dataset:

1. **Prepare your dataset**:
   - Create directories for original and Vinahouse songs:
     ```
     data/
     ├── original/
     │   ├── train/
     │   └── val/
     └── vinahouse/
         ├── train/
         └── val/
     ```
   - Place audio files (WAV format recommended) in the respective directories

2. **Configure training parameters**:
   - Edit the configuration file at `vinahouse_remixer/configs/training_config.json`
   - Adjust parameters like batch size, learning rate, etc. as needed

3. **Run the training script**:
   ```bash
   python -m vinahouse_remixer.training.train --config vinahouse_remixer/configs/training_config.json
   ```

4. **Monitor training progress**:
   - Checkpoints will be saved to the directory specified in the config file
   - Training history will be plotted and saved as an image

### Inference

#### Single File Processing

```bash
python -m vinahouse_remixer.inference.inference \
    --input path/to/your/song.wav \
    --output path/to/output/directory \
    --model path/to/pretrained/model.pt \
    --config vinahouse_remixer/configs/training_config.json \
    --style-strength 0.8
```

#### Batch Processing

```bash
python -m vinahouse_remixer.inference.inference \
    --input path/to/input/directory \
    --output path/to/output/directory \
    --model path/to/pretrained/model.pt \
    --config vinahouse_remixer/configs/training_config.json \
    --style-strength 0.8 \
    --batch \
    --file-ext .wav
```

#### Style Strength Parameter

The `--style-strength` parameter controls how strongly the Vinahouse style is applied:
- `0.0`: No style transfer (original song)
- `0.5`: Balanced mix of original and Vinahouse style
- `1.0`: Full Vinahouse style transformation

### Configuration

The configuration file (`training_config.json`) contains parameters for both training and inference:

```json
{
    "sample_rate": 44100,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 80,
    "segment_duration": 5.0,
    
    "original_train_dir": "data/original/train",
    "vinahouse_train_dir": "data/vinahouse/train",
    "original_val_dir": "data/original/val",
    "vinahouse_val_dir": "data/vinahouse/val",
    
    "paired_data": false,
    "max_train_samples": 1000,
    "max_val_samples": 200,
    
    "latent_dim": 64,
    "time_steps": 1000,
    "beta_schedule": "linear",
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    
    "batch_size": 16,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "checkpoint_dir": "models/checkpoints",
    "save_every": 5,
    
    "device": "cuda"
}
```

Key parameters:
- **Audio processing**: `sample_rate`, `n_fft`, `hop_length`, `n_mels`, `segment_duration`
- **Dataset**: `original_train_dir`, `vinahouse_train_dir`, `paired_data`, `max_train_samples`
- **Model**: `latent_dim`, `time_steps`, `beta_schedule`
- **Training**: `batch_size`, `learning_rate`, `num_epochs`, `checkpoint_dir`
- **Hardware**: `device` (cuda or cpu)

## Project Structure

```
vinahouse_remixer/
├── __init__.py
├── preprocessing/
│   └── audio_processor.py
├── style_transfer/
│   └── diffusion.py
├── rhythm_transform/
│   └── rhythm_transformer.py
├── training/
│   └── train.py
├── inference/
│   └── inference.py
├── configs/
│   └── training_config.json
├── utils/
│   └── ...
├── models/
│   └── ...
└── data/
    ├── original/
    └── vinahouse/
```

## Technical Details

### Hybrid Approach

Our system uses a hybrid approach combining:

1. **Diffusion Models for Spectral Transformation**:
   - Learns to transform the spectral characteristics of audio
   - Preserves content while transferring style
   - Based on the Riffusion architecture (adapted from Stable Diffusion)

2. **Rhythm and Tempo Transformation**:
   - Adjusts tempo to the Vinahouse range (132-142 BPM)
   - Enhances or adds four-on-the-floor kick pattern
   - Boosts bass frequencies for the characteristic Vinahouse sound

### Audio Processing Pipeline

1. **Preprocessing**:
   - Load and normalize audio
   - Extract features (tempo, beats, etc.)
   - Compute mel-spectrograms

2. **Style Transfer**:
   - Transform mel-spectrograms using diffusion model
   - Apply style conditioning

3. **Rhythm Transformation**:
   - Adjust tempo
   - Transform drum patterns
   - Enhance bass frequencies

4. **Output Generation**:
   - Convert transformed mel-spectrograms back to audio
   - Apply post-processing for quality enhancement

## Examples

### Example 1: Basic Remixing

```python
from vinahouse_remixer.remixer import VinahouseRemixer

remixer = VinahouseRemixer(model_path="models/model_final.pt")
remixer.remix("examples/original.wav", "examples/remixed.wav")
```

### Example 2: Batch Processing

```python
import os
from vinahouse_remixer.remixer import VinahouseRemixer

remixer = VinahouseRemixer(model_path="models/model_final.pt")

input_dir = "examples/originals"
output_dir = "examples/remixed"
os.makedirs(output_dir, exist_ok=True)

input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")]
remixed_files = remixer.batch_remix(input_files, output_dir)
```

### Example 3: Adjusting Style Strength

```python
from vinahouse_remixer.remixer import VinahouseRemixer

# Create remixers with different style strengths
light_remixer = VinahouseRemixer(model_path="models/model_final.pt", style_strength=0.3)
medium_remixer = VinahouseRemixer(model_path="models/model_final.pt", style_strength=0.6)
full_remixer = VinahouseRemixer(model_path="models/model_final.pt", style_strength=1.0)

# Create remixes with different intensities
light_remixer.remix("examples/original.wav", "examples/light_remix.wav")
medium_remixer.remix("examples/original.wav", "examples/medium_remix.wav")
full_remixer.remix("examples/original.wav", "examples/full_remix.wav")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
