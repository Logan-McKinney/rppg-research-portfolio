"""
test_model_unit.py
==================
Quick unit test to confirm DeepPhysMultiTask_v1 runs correctly
with and without Patrick's motion features, before touching real data.

HOW TO RUN:
    conda activate rppg-toolbox
    cd <local-path>
    python tools/test_model_unit.py

Expected output:
    [TEST 1] Without motion features (heuristic fallback)...
      out shape   : torch.Size([30, 1])
      peaks shape : torch.Size([30, 1])
      troughs shape: torch.Size([30, 1])
      PASS

    [TEST 2] With Patrick's motion features (CSV path)...
      out shape   : torch.Size([30, 1])
      peaks shape : torch.Size([30, 1])
      troughs shape: torch.Size([30, 1])
      PASS

    [TEST 3] Check for NaN in outputs...
      PASS — no NaN values

    All tests passed.
"""

import torch
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_methods.model.DeepPhysMultiTask_v1 import DeepPhysMultiTask_v1

# ─────────────────────────────────────────────
#  Config — matches your YAML settings
# ─────────────────────────────────────────────
IMG_SIZE    = 72    # matches H: 72, W: 72 in your YAML
FRAME_DEPTH = 15    # number of frames per clip fed to the model at once
BATCH_N     = 2     # number of clips in a batch (N)
# N*T total frames in the batch
N_T = BATCH_N * FRAME_DEPTH  # = 30

def run_tests():
    model = DeepPhysMultiTask_v1(img_size=IMG_SIZE, frame_depth=FRAME_DEPTH)
    model.eval()

    # Standard rPPG input: [N*T, 6, H, W]
    # 6 channels = 3 DiffNormalized + 3 Standardized (raw)
    fake_input = torch.randn(N_T, 6, IMG_SIZE, IMG_SIZE)

    # ── Test 1: no motion features — heuristic fallback ───────────────────────
    print("[TEST 1] Without motion features (heuristic fallback)...")
    with torch.no_grad():
        out, peaks, troughs = model(fake_input, motion_csv_features=None)
    print(f"  out shape    : {out.shape}")
    print(f"  peaks shape  : {peaks.shape}")
    print(f"  troughs shape: {troughs.shape}")
    assert out.shape    == (N_T, 1), f"Expected ({N_T}, 1), got {out.shape}"
    assert peaks.shape  == (N_T, 1), f"Expected ({N_T}, 1), got {peaks.shape}"
    assert troughs.shape == (N_T, 1), f"Expected ({N_T}, 1), got {troughs.shape}"
    print("  PASS\n")

    # ── Test 2: with Patrick's motion features ────────────────────────────────
    print("[TEST 2] With Patrick's motion features...")
    # Shape: [N, T, 8] — N clips, T frames each, 8 feature channels
    fake_motion = torch.randn(BATCH_N, FRAME_DEPTH, 8)
    with torch.no_grad():
        out, peaks, troughs = model(fake_input, motion_csv_features=fake_motion)
    print(f"  out shape    : {out.shape}")
    print(f"  peaks shape  : {peaks.shape}")
    print(f"  troughs shape: {troughs.shape}")
    assert out.shape    == (N_T, 1), f"Expected ({N_T}, 1), got {out.shape}"
    assert peaks.shape  == (N_T, 1), f"Expected ({N_T}, 1), got {peaks.shape}"
    assert troughs.shape == (N_T, 1), f"Expected ({N_T}, 1), got {troughs.shape}"
    print("  PASS\n")

    # ── Test 3: no NaN in outputs ─────────────────────────────────────────────
    print("[TEST 3] Check for NaN in outputs...")
    assert not torch.isnan(out).any(),    "NaN detected in rPPG output"
    assert not torch.isnan(peaks).any(),  "NaN detected in peak output"
    assert not torch.isnan(troughs).any(),"NaN detected in trough output"
    print("  PASS\n")

    print("All tests passed.")

if __name__ == "__main__":
    run_tests()
