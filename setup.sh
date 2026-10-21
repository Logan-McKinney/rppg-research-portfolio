#!/usr/bin/env bash
set -euo pipefail
# Creates a NEW environment; conda fails safely if this name already exists.
conda create -n rppg-portfolio python=3.8 pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 -c pytorch -y
echo "Run: conda activate rppg-portfolio; python -m pip install -r requirements.txt"
