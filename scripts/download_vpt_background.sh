#!/bin/bash
# Background download script for OpenAI VPT dataset
# Downloads to /data/vvm33/vpt_dataset (1TB+ storage available)
# Runs in background using nohup

set -e

VPT_DIR="/data/vvm33/vpt_dataset"
LOG_FILE="$VPT_DIR/download.log"

# According to OpenAI VPT repo, data is hosted on Azure Blob Storage
# Format: https://openaipublic.blob.core.windows.net/minecraft-vpt/
# Contains both 1080p and 720p versions

AZURE_URL="https://openaipublic.blob.core.windows.net/minecraft-vpt"

echo "Starting VPT dataset download..." | tee -a "$LOG_FILE"
echo "Target directory: $VPT_DIR" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd "$VPT_DIR"

# Download using azcopy (faster) or wget/curl as fallback
# The dataset has multiple parts, we'll download a subset first for testing

# Create subdirectories
mkdir -p "$VPT_DIR/videos"
mkdir -p "$VPT_DIR/annotations"

# Download manifest/inventory first
echo "Downloading manifest..." | tee -a "$LOG_FILE"
wget -q --show-progress "$AZURE_URL/manifest.json" -O "$VPT_DIR/manifest.json" 2>&1 | tee -a "$LOG_FILE" || true

# Download a small subset first ( contractor videos - smaller and diverse)
# According to the paper, contractor data is ~100GB vs 1TB for full dataset
CONTRACTOR_PREFIX="contractor"

echo "Downloading contractor dataset subset (for testing)..." | tee -a "$LOG_FILE"

# Use aria2c for faster parallel downloads if available, else wget
if command -v aria2c &> /dev/null; then
    echo "Using aria2c for parallel downloads..." | tee -a "$LOG_FILE"
    # Download a few contractor videos as a test
    for i in $(seq -w 1 10); do
        aria2c -x 4 -s 4 --file-allocation=none \
            "$AZURE_URL/videos/${CONTRACTOR_PREFIX}_${i}.mp4" \
            -d "$VPT_DIR/videos" 2>&1 | tee -a "$LOG_FILE" &
        aria2c -x 4 -s 4 --file-allocation=none \
            "$AZURE_URL/annotations/${CONTRACTOR_PREFIX}_${i}.jsonl" \
            -d "$VPT_DIR/annotations" 2>&1 | tee -a "$LOG_FILE" &
    done
    wait
else
    echo "Using wget (slower, install aria2c for faster downloads)..." | tee -a "$LOG_FILE"
    for i in $(seq -w 1 10); do
        wget -q --show-progress -c "$AZURE_URL/videos/${CONTRACTOR_PREFIX}_${i}.mp4" \
            -P "$VPT_DIR/videos" 2>&1 | tee -a "$LOG_FILE" &
        wget -q --show-progress -c "$AZURE_URL/annotations/${CONTRACTOR_PREFIX}_${i}.jsonl" \
            -P "$VPT_DIR/annotations" 2>&1 | tee -a "$LOG_FILE" &
    done
    wait
fi

echo "========================================" | tee -a "$LOG_FILE"
echo "Download completed at: $(date)" | tee -a "$LOG_FILE"
echo "Downloaded files:" | tee -a "$LOG_FILE"
find "$VPT_DIR" -type f -ls | tee -a "$LOG_FILE"
