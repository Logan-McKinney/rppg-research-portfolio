# Setup, data, and execution guide

## Environment

The original setup used Python 3.8 and PyTorch 1.12.1 / torchvision 0.13.1. Progress evidence specifies MediaPipe 0.10.5 with its legacy `mp.solutions.face_mesh` API. Newer MediaPipe versions may remove/change that API. The setup script creates `rppg-portfolio` without deleting another environment. Run it only if establishing that historical environment is appropriate for your platform.

The retained requirements contain historical pins (including h5py 2.10.0, SciPy 1.5.2, NumPy 1.22.0), plus utilities imported but omitted from the original list: PyWavelets, natsort, fastdtw, dtw and IPython. Installation and combined compatibility have not been validated; do not interpret these as a solved dependency lock. A modern environment passing model-only checks does not establish compatibility of the entire toolbox. No framework installation is required to review the code, results or diagrams.

Use the repository root as the working directory; the Haar detector path is relative to it. Set `DEVICE` and `NUM_OF_GPU_TRAIN` for the target machine. The retained models and trainers have differing DataParallel assumptions; CPU inference of all trainers is not verified by model-only tests.

## Dataset access and layouts

Request access directly from dataset owners and follow their conditions. No raw data, individual BVP measurements or cached clips are included. Local paths are examples; `data/`, `cache/`, weights and generated outputs are ignored by Git.

| Dataset | Source / original citation | Loader |
| --- | --- | --- |
| PURE | R. Stricker, S. Müller, H.-M. Gross, *Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot*, RO-MAN 2014, pp. 1056–1062. [Owner page](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure). | `PURELoader.py` |
| UBFC-rPPG | S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, *Unsupervised skin tissue segmentation for remote photoplethysmography*, Pattern Recognition Letters. [Owner page](https://sites.google.com/view/ybenezeth/ubfcrppg). The inherited README gave 2017; retain that historical attribution pending bibliography verification. | `UBFCLoader.py` |
| SCAMPS | D. McDuff, M. Wander, X. Liu, B. Hill, J. Hernandez, J. Lester, T. Baltrusaitis, *SCAMPS: Synthetics for Camera Measurement of Physiological Signals*, 2022. [Paper](https://arxiv.org/abs/2206.04197) · [Project](https://github.com/danmcduff/scampsdataset). | `SCAMPSLoader.py` |
| MMPD | J. Tang, K. Chen, Y. Wang, Y. Shi, S. Patel, D. McDuff, X. Liu, *MMPD: Multi-Domain Mobile Video Physiology Dataset*, 2023. [Project](https://github.com/THU-CS-PI/MMPD_rPPG_dataset). | `MMPDLoader.py` |
| COHFACE | Existing loader retained; request and verify access with the dataset owner before use. No COHFACE result is claimed here. | `COHFACELoader.py` |
| Daily / Aya | Internal collaborative dataset named “Daily” in progress reports and “Aya” in code, with restricted availability. No verified public citation or public access mechanism is established; this repository provides no dataset access or redistribution rights. | `AyaLoader.py` |

The shared README said only three datasets were supported while also listing MMPD and future COHFACE support. The current snapshot contains the six loaders above; their presence does not prove validated experiment coverage for each dataset.

```text
data/PURE/
  01-01/
    01-01/                  PNG frames
    01-01.json              reference waveform metadata
  01-02/ ...

data/UBFC/
  subject1/
    vid.avi
    ground_truth.txt
  subject2/ ...

data/SCAMPS/
  Train/P00001.mat ...
  Val/P00001.mat ...
  Test/P00001.mat ...

data/MMPD/
  subject1/p1_0.mat ... p1_19.mat
  subject2/p2_0.mat ...

data/Daily/
  <subject>/
    vid_<distance>_<lux>_<motion>_<angle>_<device>/
      <timestamp_ms>.png
    gt_<distance>_<lux>_<motion>_<angle>_<device>/
      pulse_data.csv        PC_Timestamp_ms, Signal_Value columns
```

Daily motion loading currently expects session-prefixed pose/lips/left_eye/right_eye CSVs in the subject directory, one level above the video session directory. Its docstring shows an older in-session layout; the active code is authoritative for the current interface. Do not publish those files.

## Configuration parameters

| Parameter | Meaning / caution |
| --- | --- |
| `TOOLBOX_MODE` | `train_and_test`, `only_test`, or `unsupervised_method`. Inference requires a separately supplied `INFERENCE.MODEL_PATH`. |
| `DATA_PATH` / `CACHED_PATH` | Raw-data root / preprocessed output root; replace examples before running. Generated file lists contain local clip paths and stay out of Git. |
| `DO_PREPROCESS` | Enable for the first pass; disable to reuse compatible caches. |
| `EXP_DATA_NAME` | Empty uses a generated name from config; an explicit value overrides it. Check cache separation manually for new motion/ROI experiments. |
| `BEGIN` / `END` | Fractional split used by the standard loaders. The Daily loader instead uses explicit subject lists and condition filters in its source. Do not assume one split policy across datasets. |
| `DATA_TYPE` / `LABEL_TYPE` | Input/label transforms; DeepPhys-style research models expect six channels from difference and appearance streams. Historical recipes often use `DiffNormalized` + `Standardized`, not literal unprocessed RGB. |
| `DATA_FORMAT` | Research trainer expects batched `NDCHW` and flattens N×D for frame processing. Other formats exist but are not interchangeable here. |
| `DO_CHUNK` / `CHUNK_LENGTH` | Split into frame windows, typically 180 at 30 fps or 300 at 50 fps in the documented work. Multitask trainer passes this depth to its model. |
| `ROI_METHOD` / `ROI_REGION` | `haar` or `mediapipe`; glabella, forehead, left/right cheek, nose, full face. Only documented glabella/Haar experiments establish observed behavior. |
| `CROP_FACE` | Enable crop behavior. Legacy flat/nested config forms coexist in the original recipes; review compatibility for the chosen loader. |
| `DYNAMIC_DETECTION` | Haar box recomputation versus reusing a detection. |
| `DYNAMIC_DETECTION_FREQUENCY` | Frames between dynamic detections. |
| `LARGE_FACE_BOX` / `LARGE_BOX_COEF` | Enlarge Haar box where supported; values preserved from source. |
| `H` / `W` | Resized ROI dimensions; 72×72 in key research recipes. Models contain fixed spatial assumptions. |
| `MODEL.NAME` | Selects model/trainer routing; names are case-sensitive. |
| `TEST.USE_LAST_EPOCH` | `True` uses last epoch; `False` enables the trainer's best-validation checkpoint behavior. Inspect each retained trainer. |
| `TEST.METRICS` | MAE, RMSE, MAPE, Pearson, waveform MSE/PSNR/Correlation, SNR, DTW, peak metrics as implemented. |
| `UNSUPERVISED.METHOD` | Classical options include ICA, POS, CHROM, GREEN, LGI, PBV. |

## Preserved execution examples

1. **PURE → UBFC, TSCAN:** prepare PURE and UBFC layouts, edit `configs/train_configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml`, then run `python main.py --config_file configs/train_configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml`. This inherited recipe uses 80% PURE training and 20% validation; it is not the shared README's 70% UBFC ablation.
2. **SCAMPS → UBFC, DeepPhys:** prepare SCAMPS splits and UBFC, edit `configs/train_configs/SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml`, then run the same entry point with that config. The inherited description uses an 80%/20% SCAMPS training/validation setting; inspect loader paths and actual config before use.
3. **Inference:** use `configs/infer_configs/PURE_UBFC_TSCAN_BASIC.yaml` and set a compatible checkpoint. The old README referred to `PURE_PURE_UBFC_TSCAN_BASIC.yaml` under inference, but that filename does not exist there; this corrected reference points to the retained file.
4. **Unsupervised UBFC:** edit `configs/infer_configs/UBFC_UNSUPERVISED.yaml`, then run `python main.py --config_file configs/infer_configs/UBFC_UNSUPERVISED.yaml`. No trained neural checkpoint is needed, but authorized raw data is.

These are execution instructions, not claims that training, dataset loading or inference were run during preparation. See the validation report for actual scope.

## Motion feature interface

`tools/run_motion_extraction.py` is the PURE batch adapter using Patrick's MediaPipe feature helpers. Review its constants for input/output roots and the external Face Landmarker task path. The task asset and its rights are separate; no weights are bundled. PURELoader looks for `data/motion/PURE/<subject-trial>_motion.csv`. It fills missing values, pads/trims rows and saves chunks. BaseLoader loads sibling `*_motion<index>.npy`, applies per-clip normalization, and returns motion at batch index four.

The expected order is `[head_tx_vel, head_ty_vel, lips_vel, yaw, pitch, roll, lips_accel, eor_l]`. Daily uses derived proxy channels from different CSV schemas; identical names do not establish identical physical measurements. The last channel is a true landmark openness ratio in the PURE extractor and a velocity-derived proxy in Daily. See Results for channel/alignment caveats.

`DeepPhysMultiTask_v1.forward` accepts motion shaped `(N,T,8)`. Without an optional tensor it computes heuristic auxiliary features; a loader-supplied zero tensor bypasses that heuristic path. The existing synthetic test checks both direct-call paths without participant data.

## Adding a loader

Add a subclass under `dataset/data_loader/`, implement raw-data discovery/splitting, preprocessing and `read_video`/`read_wave` as appropriate, and register it where loaders are selected in `main.py`. Keep cache saving/loading and sample semantics compatible with BaseLoader, including the motion element where used. Define new config keys in `config.py` before using them in YAML. Avoid changing `__len__`, sample return structure or save/load interfaces without checking every trainer.
