#!/bin/bash
set -e   # exit immediately on any error

# ──────────────────────────────────────────────────────────────────
# 0. Configuration — EDIT these two lines before submitting
# ──────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/GarvCosmo/GNR638-Project.git"   # ← replace
REPO_DIR="GNR638-Project"                                        

# ──────────────────────────────────────────────────────────────────
# 1. Clone repository (internet is available during setup)
# ──────────────────────────────────────────────────────────────────
echo "==> Cloning repository …"
git clone "$REPO_URL"
cd "$REPO_DIR"

# ──────────────────────────────────────────────────────────────────
# 2. Create conda environment
# ──────────────────────────────────────────────────────────────────
echo "==> Creating conda environment gnr_project_env (Python 3.11) …"
conda create -n gnr_project_env python=3.11 -y

# Activate — works in both login and non-login shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnr_project_env

# ──────────────────────────────────────────────────────────────────
# 3. Install Python dependencies
# ──────────────────────────────────────────────────────────────────
echo "==> Installing Python packages …"
pip install --upgrade pip
pip install \
    numpy \
    pandas \
    opencv-python-headless \
    pillow \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    transformers>=4.45.0 \
    accelerate \
    qwen-vl-utils \
    sentencepiece \
    huggingface_hub

# ──────────────────────────────────────────────────────────────────
# 4. Download model weights (offline during inference)
# ──────────────────────────────────────────────────────────────────
echo "==> Downloading Qwen2-VL-7B-Instruct weights …"
python - <<'EOF'
from huggingface_hub import snapshot_download
import os

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights", "Qwen2-VL-7B-Instruct")
os.makedirs(save_dir, exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen2-VL-7B-Instruct",
    local_dir=save_dir,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print(f"Model saved to: {save_dir}")
EOF

echo ""
echo "==> Setup complete. Run with:"
echo "    conda activate gnr_project_env"
echo "    python inference.py --test_dir <absolute_path_to_test_dir>"
