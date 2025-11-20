#!/usr/bin/env bash
set -e

############################
# 0. Basic system setup
############################

sudo apt-get update

# Install Python & pip if not already present
sudo apt-get install -y python3 python3-pip

# Install unzip and other helpers
sudo apt-get install -y unzip

# Install Kaggle CLI
pip3 install --user kaggle

# Ensure ~/.local/bin is on PATH (where pip --user installs kaggle)
export PATH=$PATH:~/.local/bin

############################
# 1. Configure Kaggle API
############################

# Create kaggle directory if it doesn't exist
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle

echo ">>> Now you need to upload kaggle.json to the VM."
echo ">>> In a separate terminal on your local machine, run something like:"
echo "    gcloud compute scp kaggle.json YOUR_VM_NAME:~/.kaggle/kaggle.json --zone=YOUR_ZONE"
echo ">>> Once that is done, press ENTER here to continue."
read

chmod 600 ~/.kaggle/kaggle.json

############################
# 2. Create destination directory
############################

IMAGENET_DIR=~/imagenet
mkdir -p "$IMAGENET_DIR"
cd "$IMAGENET_DIR"

############################
# 3. Download competition data
############################

# Kaggle competition: imagenet-object-localization-challenge
echo ">>> Downloading ImageNet Object Localization Challenge data..."
kaggle competitions download -c imagenet-object-localization-challenge

echo ">>> Download complete. Extracting zip files..."

# Unzip every zip file in IMAGENET_DIR
for z in *.zip; do
    echo "Unzipping $z ..."
    unzip -q "$z" -d "$IMAGENET_DIR"
done

echo ">>> All files downloaded and extracted to $IMAGENET_DIR"