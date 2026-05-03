#!/bin/bash
# EdgeSR Full Pipeline Script
# Run on the A800 server (no internet access).
# Usage: bash scripts/pipeline.sh [baseline|edgesr]

set -e

MODEL=${1:-edgesr}
DATA_ROOT=${DATA_ROOT:-./data}
CONFIG=${CONFIG:-configs/default.yaml}
DEVICE=${DEVICE:-cuda}

echo "========================================="
echo " EdgeSR Pipeline - Model: $MODEL"
echo "========================================="

# Step 1: Verify environment
echo ""
echo "[1/5] Verifying environment..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Step 2: Verify dataset exists
echo ""
echo "[2/5] Verifying dataset..."
python -c "
import os
for d in ['$DATA_ROOT/DIV2K', '$DATA_ROOT/Flickr2K', '$DATA_ROOT/benchmark']:
    exists = os.path.exists(d)
    print(f'  {d}: {\"OK\" if exists else \"MISSING\"}')"

# Step 3: Train model
echo ""
echo "[3/5] Training $MODEL model..."
python src/train.py --config "$CONFIG" --model "$MODEL" --device "$DEVICE"

# Step 4: Evaluate
echo ""
echo "[4/5] Evaluating $MODEL model..."
python src/test.py --config "$CONFIG" --model "$MODEL" --checkpoint ./checkpoints/best.pt --device "$DEVICE"

# Step 5: Generate sample results
echo ""
echo "[5/5] Generating sample results..."
python src/inference.py --config "$CONFIG" --model "$MODEL" --checkpoint ./checkpoints/best.pt --input ./data/benchmark/Set5 --output ./results/$MODEL --device "$DEVICE"

echo ""
echo "========================================="
echo " Pipeline complete!"
echo "========================================="
