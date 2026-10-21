# Logan's documented experiments and investigations

These values come from private progress presentations, cross-checked against available code. They are not independently recomputed results. Internal decks and participant-level outputs are excluded; evidence IDs refer to [PROVENANCE.md](PROVENANCE.md).

## PURE ROI and motion comparison

E06 reports 10 CPU epochs on PURE, with the same train/test split for Haar baseline and MediaPipe glabella + real motion CSVs. The surviving glabella/motion recipe uses 30 fps, 180-frame clips, 72×72 crops, and train/validation/test ranges 0–0.7 / 0.7–0.8 / 0.8–1.0. The exact run-to-config mapping is unverified; the slides do not establish a clean one-factor ablation.

| Metric | Haar baseline (as labeled in E06) | DeepPhysMultiTask_v1, glabella + motion |
| --- | ---: | ---: |
| HR Pearson r | 0.9997 | 0.76 |
| HR MAE (bpm) | 0.22 | 4.03 |
| Waveform correlation | 0.633 | 0.478 |
| Waveform SNR (dB) | 1.34 | -0.18 |
| DTW distance | 9.17 | 9.69 |

The baseline is stronger in this reported comparison. The high HR correlation is not equivalent to perfect waveform recovery. The exact baseline model/checkpoint and split leakage protections are not independently established by retained run artifacts. The proposed matched Haar+motion run would isolate a different question; it is not presented as completed.

E04 also reports a five-epoch qualitative ROI check: both loss-only ROI choices and the Haar combined run tracked peaks, while the combined MediaPipe run remained flat. This does not supply a quantitative ROI ranking across all regions.

## Daily/Aya Nodding experiments

E07–E09 report a local subset: Nodding, 70 lux, distances 1/2/3 m, 50 fps, 300-frame clips (six seconds), batch size two, CPU. The loader uses seven train and three test subject IDs; slides report 241 training chunks and 108 test chunks. Raw IDs, recordings and signals are not bundled. The 30 motion-extraction sessions correspond to three distances per included subject, not 30 different people. Access conditions and official dataset identity require team confirmation.

| Run | Epochs | Main documented change | HR Pearson | HR MAE (bpm) | Waveform r | DTW |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 3 | Temporal-depth bug and weighted loss present | -0.198 | 12.70 | 0.0084 | 26.98 |
| 2 | 10 | Depth set to 300; weighted loss removed | 0.240 | 5.94 | 0.0262 | 18.59 |
| 3 | 20 | SNR weight 0.13 → 0.01; wavelet level 1 | -0.297 | 38.49 | 0.0310 | 15.99 |
| 4 | 20 | Wavelet processing per chunk | -0.238 | 17.33 | 0.0492 | 16.28 |
| 5 | 20 | v1 motion integration, old channels and per-clip scaling | -0.734 | 21.57 | 0.0485 | 15.81 |

Run 2 has the best reported HR Pearson and MAE; Run 4 has the highest waveform correlation; Run 5 has the lowest DTW in this five-run table while regressing in HR accuracy. The E08 slide says DTW improved monotonically, but its own table increases from Run 3 to Run 4; that interpretation is not repeated here. Changing multiple settings and epoch counts prevents attributing every difference to one fix.

E07's earlier five-epoch baseline reduced loss from 0.753 to 0.742 with flat waveform output. E08 reports Run 4 total loss falling 2.923 → 2.361 and PPG loss 0.784 → 0.645. Loss convergence did not establish reliable cardiac-frequency recovery.

## Other collapse investigation

E10 reports a 60-epoch run whose predictions were nearly flat (standard deviation 0.016), despite train loss decreasing approximately 0.81 → 0.42. Its displayed MAE 7.43 bpm, RMSE 11.30 bpm, MAPE 13.70%, HR Pearson 0.843, waveform MSE 1.906 and waveform correlation 0.047 are explicitly from a **one-epoch sanity run**, not final 60-epoch evaluation. Dataset/run identity is insufficiently specified for a benchmark claim; final 60-epoch metrics were suppressed by invalid metric warnings.

## Completed versus exploratory

Code and slides support temporal-depth correction, identity initialization, motion-batch routing, per-clip scaling, chunk-wise label processing, signal/gradient diagnostics and Daily nearest-neighbor reference alignment. They do not establish a successful final fix for all waveform-collapse cases.

Future directions include fixed/global motion scaling, smoother motion channels, a controlled Haar+motion experiment, 28-ROI signal-quality fusion, SQI gating, additional motion conditions, and formal flexible tensor contracts. The Daily channel correction is claimed in E09 but absent in the retained source; it remains a discrepancy, not an implemented portfolio feature. No predicted future improvement is presented as a measured outcome.
