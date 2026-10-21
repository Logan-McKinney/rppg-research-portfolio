# Release review

Prepared October 21, 2026 for a new **private** portfolio repository. Code preparation and redistribution of the seven original team figures were confirmed by Logan. The original research directories and private repository were read only.

## Included and excluded

Included: one selected implementation, four research models and trainers, inherited comparison models, loaders, preprocessing/ROI code, YAML recipes, evaluation/classical baselines, PURE motion batch adapter, existing synthetic model check, full reported tables, approved figures, source hashes, attribution and upstream license.

Excluded: source Git history, private presentation decks, raw video and participant images, individual signals, cached data and motion arrays/CSVs, pickle exports/notebook outputs, research checkpoints, Face Landmarker task weights, logs, IDE files, exploratory waveform-combination scripts, and standalone 3DDFA/CropSense repositories. Patrick's standalone tracker is not copied: the useful integration adapter is retained with credit.

## Remaining factual and provenance questions

1. Confirm the historical upstream toolbox revision and intended distribution terms for team additions, including Patrick-derived helpers. The recovered upstream license is preserved; no blanket new license is invented.
2. Confirm exact original UBFC → PURE evaluation coverage, configs, model revisions, run-to-figure mappings and checkpoint selection if stronger reproducibility claims are desired. Current defaults cannot establish these retrospectively.
3. Locate the later AyaLoader version containing the yaw/pitch derivative change claimed in E09, or confirm the slide described a local change not present in the supplied snapshot. Current eye-gradient code is preserved with an explicit discrepancy note.
4. Confirm the official Daily/Aya dataset citation and access conditions before inviting others to obtain/use it. No data is distributed here.

## Public-release status

The repository is prepared for **private review**. It should remain private until Logan reviews the credits and confirms the historical/team licensing points above. Figure permission, Matthew Dowell's role, and Dr. Tianming Zhao's advising role were confirmed during preparation. Exact historical reproducibility is limited and is disclosed in the README; resolving every old experiment is not required to present the work truthfully, but the gaps must remain visible.

See [VALIDATION.md](VALIDATION.md) for executed checks, outcomes and deliberate limits. No clinical validation or full environment recreation is claimed.
