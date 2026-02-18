#!/bin/bash
# Download OpenAI VPT Contractor Demonstrations
# 
# From README: Data is at openaipublic.blob.core.windows.net/minecraft-rl/
# Structure: data/10.0/ contains .mp4 and .jsonl files with same basename
#
# Example pairing:
#   cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4
#   cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl

set -e

VPT_DIR="/data/vvm33/vpt_contractor"
mkdir -p "$VPT_DIR"
cd "$VPT_DIR"

echo "=========================================="
echo "Downloading OpenAI VPT Contractor Data"
echo "Target: $VPT_DIR"
echo "=========================================="

# Base URL for contractor data (from README)
BASE_URL="https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0"

# Sample file from README (for testing)
SAMPLE_VIDEO="cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4"
SAMPLE_JSONL="cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl"

echo ""
echo "Testing with sample file from README..."
echo "Downloading: $SAMPLE_VIDEO"

if wget -q --timeout=30 --show-progress "$BASE_URL/$SAMPLE_VIDEO" -O "$SAMPLE_VIDEO" 2>&1; then
    echo "✅ Video downloaded successfully"
    
    echo "Downloading: $SAMPLE_JSONL"
    if wget -q --timeout=30 --show-progress "$BASE_URL/$SAMPLE_JSONL" -O "$SAMPLE_JSONL" 2>&1; then
        echo "✅ Annotation downloaded successfully"
        
        # Check the JSONL structure
        echo ""
        echo "Sample annotation structure:"
        head -1 "$SAMPLE_JSONL" | python3 -m json.tool | head -30 || head -1 "$SAMPLE_JSONL"
    else
        echo "❌ Failed to download annotation"
    fi
else
    echo "❌ Failed to download video"
    echo "The URL may have changed or require authentication"
    exit 1
fi

# Get the file size
VIDEO_SIZE=$(du -h "$SAMPLE_VIDEO" | cut -f1)
JSONL_SIZE=$(du -h "$SAMPLE_JSONL" | cut -f1)

echo ""
echo "Downloaded:"
echo "  Video: $SAMPLE_VIDEO ($VIDEO_SIZE)"
echo "  Annotation: $SAMPLE_JSONL ($JSONL_SIZE)"

# Now try to download more contractor files
echo ""
echo "Attempting to download more contractor recordings..."
echo "(This may take a while - each video is several GB)"

# Common contractor naming patterns
# From the README, contractors have UUID-based filenames

# Try downloading a few more samples
for i in $(seq 1 5); do
    echo ""
    echo "Looking for additional contractor files ($i/5)..."
    
    # Try to list directory (may not work)
    # Instead, try common patterns or download index if available
    
    # For now, just verify we can access the sample
    if [ -f "$SAMPLE_VIDEO" ]; then
        echo "✅ Sample file is accessible"
        break
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Downloaded files:"
ls -lh "$VPT_DIR"
echo ""
echo "To download more contractor data:"
echo "1. Check the index files at:"
echo "   $BASE_URL/"
echo "2. Or use azcopy for bulk download:"
echo "   azcopy copy '$BASE_URL/*' '$VPT_DIR' --recursive"
echo ""
echo "Data location: $VPT_DIR"
echo "=========================================="
