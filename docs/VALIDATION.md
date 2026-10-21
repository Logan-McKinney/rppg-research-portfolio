# Validation record — October 21, 2026

## Executed checks

| Check | Outcome |
| --- | --- |
| Python syntax via `ast.parse` | All 50 selected Python files parsed. One inherited invalid-escape warning in UBFCLoader's regex is nonfatal and retained. |
| YAML parsing via `yaml.safe_load` | All 48 configuration files parsed after repairing one original truncated `DATA_FO` entry. |
| Existing `tools/test_model_unit.py` | All three tests passed on CPU: v1 heuristic fallback, synthetic eight-channel motion input, and output NaN checks. The script also asserts waveform/peak/trough output shapes `(30,1)`. |
| Runtime for the synthetic test | Existing Python 3.13 environment, PyTorch 2.6.0+cpu; two Torch CPU threads. No framework installed. This is not validation of the historical dependency stack. |
| Documentation references | Relative Markdown file/image links checked for target existence, including all seven original figures and documented configuration paths. External citations were inspected selectively; dataset site availability was not comprehensively tested. |
| Source provenance | SHA-256 register covers 100 selected source/config/utility files plus seven original figures; final hashes verified against the publication copy. Original source hashes rechecked without changing originals. |
| Publication content scan | File types, size, machine paths, credential/token patterns, private-key headers and binary inventory reviewed. No raw participant recordings, measurement arrays, motion exports, decks, weights or environment secrets included. Pattern scanning is not an absolute guarantee. |
| Figures | Original approved PNGs visually reviewed. No visible faces or subject identifiers in retained diagram/plots. Face-containing slide screenshot excluded. |

The first test launch used the parent working directory and failed to locate the script. It was rerun from the repository root and passed; no failure is concealed as a passing model test. The initial YAML check correctly caught the inherited malformed config. Local documentation validation also caught a report link before the report existed; final validation was rerun after completing documentation.

## Scope limits

No neural training, raw dataset download, signal reconstruction on participant data, checkpoint evaluation, full runtime import/setup test or reproduction of reported research results was performed. The existing synthetic check verifies model interfaces on random tensors, not model accuracy, frame/label alignment, train/validation behavior or all ROI regions. No claim is made that the historical requirements install cleanly.

The source algorithms, current loss inconsistencies and disclosed motion/alignment limitations were preserved. The publication copy receives fresh Git history and is reviewed for excluded file types and credential/path patterns before pushing. Git object scans use all objects reachable through the fresh repository history; commit identity uses a GitHub noreply address. The original private repository receives no commits, pushes or visibility changes.
