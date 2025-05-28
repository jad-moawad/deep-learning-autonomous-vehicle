#!/bin/bash
# Download SuperTuxKart driving dataset

echo "Downloading SuperTuxKart driving dataset..."

# Create data directory
mkdir -p data

# Download dataset
if [ ! -f "data/drive_data.zip" ]; then
    wget https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip -P data/
    echo "Download complete!"
else
    echo "Dataset already downloaded."
fi

# Extract dataset
if [ ! -d "data/drive_data" ]; then
    echo "Extracting dataset..."
    unzip -q data/drive_data.zip -d data/
    echo "Extraction complete!"
else
    echo "Dataset already extracted."
fi

# Clean up
if [ -f "data/drive_data.zip" ]; then
    rm data/drive_data.zip
fi

echo "Dataset ready at data/drive_data/"