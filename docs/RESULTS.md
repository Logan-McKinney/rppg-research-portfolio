# Previously documented team results

Source: the shared research README under “Cross-Dataset Evaluation (UBFC → PURE)” and “Quantitative Results & Ablation Study.” Its reported condition is training on 70% of UBFC-rPPG and testing on PURE. These are team-reported experimental values, not a new benchmark or Logan's independently established results.

| Metric | TS_CAN_v1 | TS_CAN_Adam | DeepPhysMultiTask | DeepPhysMultiTask_v1 |
| --- | ---: | ---: | ---: | ---: |
| FFT HR MAE (bpm) | 11.76 | 5.79 | 4.28 | 4.72 |
| FFT HR RMSE (bpm) | 27.55 | 17.95 | 12.06 | 14.02 |
| HR Pearson r | 0.23 | 0.59 | 0.79 | 0.73 |
| SNR (dB) | -0.28 | 0.25 | -0.28 | -0.04 |
| Waveform correlation ρ | 0.47 | 0.53 | 0.47 | 0.50 |
| FFT MAPE | 24.26 | 11.89 | 8.77 | 9.37 |
| Waveform MSE | 1.07 | 0.94 | 1.07 | 1.01 |
| Waveform PSNR | 19.12 | 19.65 | 19.12 | 19.36 |
| Average DTW distance | 9.98 | 9.44 | 10.08 | 9.76 |
| Peak shape score | 0.53 | 0.47 | 0.53 | 0.50 |
| Average peak amplitude distance | 1.29 | 1.30 | 1.09 | 1.00 |

Lower MAE/RMSE favors the loss-only model. Higher SNR and waveform correlation favor TS_CAN_Adam. Combined processing is not universally best. The source does not provide variability estimates, significance tests or exact run identifiers; none are invented here. No numeric trough-distance result was supplied in that table, despite evaluator support for the metric.

## Evaluation implementation

[metrics.py](../evaluation/metrics.py) reconstructs/evaluates signals and reports FFT or peak-derived HR errors. `Pearson` is correlation between predicted/reference heart rates; `Correlation` uses concatenated waveforms. [post_process.py](../evaluation/post_process.py) contains detrending, integration where applicable, filtering and HR estimation. Sampling rates and label transformations depend on config.

The shared README called HR Pearson “frequency domain correlation”; the values are unchanged here, while the label is clarified using the evaluator. SNR uses reference-signal power divided by residual power. PSNR uses the reference maximum and MSE. DTW and peak scores have implementation-specific aggregation/alignment choices; compare runs only under matching preprocessing and evaluation code. Negative or undefined signal/peak metrics are possible. Training SNR loss and final reported SNR should not be conflated.

## Conditions and reproducibility gaps

1. **PURE coverage:** the shared README describes PURE testing broadly. The retained UBFC baseline/post-processing configs set PURE `BEGIN: 0.8`, `END: 1.0`. Preserve the original report; verify actual subject-trial coverage from the original experiment logs before claiming full-dataset evaluation.
2. **Filename versus content:** the v1 PURE config's name contains UBFC, but its active test dataset is PURE. The Daily filename says baseline, but active settings select `DeepPhysMultiTask_v1` and `only_test`. They are not silently renamed into different experimental conditions.
3. **Current versus historical architecture:** the team's four-way table predates Logan's 2026 integration. Current combined affine denoising differs from subtractive TS_CAN_Adam. The existing architecture figure includes intended luminance/distance features not fully delivered by the retained motion pipeline. No exact code/config/checkpoint mapping for the table is available.
4. **Loss differences:** the current shared multitask training loss is `3*PPG + 4*peak + 4*trough + 0.01*SNR`. Validation retains a different SNR weight (`0.13`). The TS_CAN baseline/post-processing trainers use waveform loss in training while retaining multi-loss validation code. These differences are preserved and prevent treating current defaults as a controlled reconstruction of the original ablation.
5. **Daily channel mismatch:** E09 says channels 0/1 were changed to yaw/pitch derivatives. Active AyaLoader still uses left-eye centroid gradients. Source and presentation therefore disagree. Per-clip standardization also remains; fixed/global normalization has not been established as completed.
6. **Alignment limitation:** Daily reference-signal alignment uses nearest neighbors after trimming and filtering video frames. Motion rows are truncated/padded then chunked; this alone does not establish timestamp alignment with the filtered frame selection. A dataset-backed alignment audit remains necessary for a defensible replication.
7. **Fallback semantics:** BaseLoader returns zeros for missing motion arrays; the model's heuristic extractor is used only when the optional argument is `None`. Missing files in a dataloader run therefore differ from a deliberate heuristic-feature experiment.
8. **Input and ROI assumptions:** models assume two three-channel streams and supported fixed spatial sizes. Several region index lists and the shared crop heuristic require anatomical validation beyond the reported glabella tests. No general flexible-input architecture is claimed.
9. **Unavailable artifacts:** checkpoints, individual signals and raw logs are deliberately excluded. Figure subject/run mappings, repeated seeds, original evaluation version and exact model selection remain unverified. Historical dependency pins are not a tested lockfile.

The [Logan experiment summary](LOGAN_EXPERIMENTS.md) is kept separate from these historical team results. The ISVC papers have their own experimental settings and reported outcomes; their published numbers are not substituted for the README's ablation table.
