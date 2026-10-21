# Validation and reproducibility

The existing `tools/test_model_unit.py` passed all three CPU checks on October 21, 2026: heuristic auxiliary fallback, supplied eight-channel motion input, and output NaN checks, with waveform/peak/trough shapes `(30,1)`. The environment was Python 3.13 and PyTorch 2.6.0+cpu, using two Torch CPU threads. This checks model interfaces on synthetic tensors, not research accuracy or the historical environment. No dependencies were installed.

Preparation checks parsed all 50 Python modules and 48 YAML configurations. The inherited UBFCLoader regex emits a nonfatal invalid-escape warning. Final documentation cleanup repeated local Markdown/image link checks, the seven-figure inventory, source/configuration/figure hash checks, and tracked-file and reachable Git-history scans for credentials, local paths and excluded participant data/assets. No source or experimental settings were changed during that cleanup; the synthetic model test was not repeated.

[source-manifest.json](source-manifest.json) retains before/after SHA-256 hashes for 100 selected source/configuration/utility files and seven approved figures, so readers can distinguish preserved research from the documented preparation changes. The complete source history, participant recordings, physiological signals, caches, motion exports, decks and weights are not distributed.

No training, participant-data evaluation, checkpoint replication, full runtime/setup test or historical result reproduction was performed. The historical requirements are not a verified dependency lock. See [Results](RESULTS.md) for evaluation, alignment and source/presentation discrepancies.
