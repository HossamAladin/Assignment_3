# Assignment 3 - SeqTrack Setup, Training, and Checkpoint Management

This repository contains the implementation for Assignment 3, which focuses on setting up, training, and managing checkpoints for the SeqTrack model on the LaSOT dataset.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, but recommended)
- Conda environment

## Setup Instructions

1. Clone the VideoX repository with sparse checkout for SeqTrack:
```bash
git clone --no-checkout https://github.com/microsoft/VideoX.git
cd VideoX
git sparse-checkout init --cone
git sparse-checkout set SeqTrack
git checkout
cd SeqTrack
```

2. Set up the environment:
```bash
# Copy assignment files to the SeqTrack directory
cp -r /path/to/assignment_files/* .

# Set up the Conda environment
bash setup_env.sh

# Activate the environment
conda activate seqtrack
```

3. Set Hugging Face token for checkpoint upload:
```bash
export HF_TOKEN=your_huggingface_token
```

## Running the Code

1. Load and filter the dataset:
```bash
python dataset_loader.py --classes airplane bicycle --output dataset_summary.txt
```

2. Train the model:
```bash
python seqtrack_train.py --classes airplane bicycle --epochs 5 --seed 8 --patch_size 8 --workers 1
```

## Directory Structure

```
SeqTrack/
  ├── seqtrack_train.py      # Main training script
  ├── dataset_loader.py      # Dataset loading and filtering
  ├── training_log.txt       # Training logs
  ├── checkpoints/           # Model checkpoints
  │   ├── epoch_1.ckpt
  │   ├── epoch_2.ckpt
  │   └── ...
  ├── requirements.txt       # Generated pip requirements
  ├── environment.yml        # Conda environment file
  ├── dataset_summary.txt    # Dataset statistics
  ├── assignment_3.md        # Assignment report
  └── setup_env.sh           # Environment setup script
```

## Key Features

- Deterministic training with seed = 8
- Training on exactly two LaSOT classes
- Checkpoint saving and automatic upload to Hugging Face
- Detailed logging with time estimates
- Full integration with the official SeqTrack implementation
