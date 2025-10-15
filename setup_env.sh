#!/bin/bash
set -e

# Create and activate conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml
echo "Conda environment 'seqtrack' created."
echo "To activate: conda activate seqtrack"

# Make sure we're in the SeqTrack directory
if [ ! -d "lib" ]; then
    echo "Error: SeqTrack directory structure not found. Please run this script from the SeqTrack directory."
    exit 1
fi

# Install any additional requirements from the SeqTrack repo
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements from requirements.txt..."
    pip install -r requirements.txt
fi

# Create directories for checkpoints and logs
mkdir -p checkpoints
mkdir -p logs

# Generate requirements.txt in assignment_3 directory
echo "Generating requirements.txt..."
mkdir -p assignment_3
pip freeze > assignment_3/requirements.txt

echo "Environment setup complete!"
echo "Please activate the environment with: conda activate seqtrack"
echo ""
echo "To run the training:"
echo "1. First load and filter the dataset:"
echo "   python dataset_loader.py --classes airplane bicycle"
echo ""
echo "2. Then train the model:"
echo "   python seqtrack_train.py --classes airplane bicycle --epochs 5 --seed 8 --patch_size 8 --workers 1"
echo ""
echo "3. For a quick test with 1 epoch:"
echo "   python seqtrack_train.py --classes airplane bicycle --epochs 1 --seed 8 --patch_size 8 --workers 1"