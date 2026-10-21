# Research provenance and contribution evidence

## Source selection

The primary 2026 research directory was selected as the canonical implementation. Its Git HEAD was `188eda7a6f02759a13115da4838aab3ceba59dfa`, matching the existing private repository's latest commit when inspected on October 21, 2026. Files were copied from the working directory, not assumed identical to the commit. [source-manifest.json](source-manifest.json) records SHA-256 hashes of every selected source file before and after preparation, including uncommitted source changes if present. Related older versions were not merged into a second implementation.

Read-only local history and GitHub inspection showed the initial toolbox upload (`43bc712`, February 2, 2026), an ablation README update (`c748930`, same date), later figure additions, and Logan's April 24 upload (`188eda7`, “First push for rppg_toolbox_motion for Logan work”). Earlier visible commits were authored under `wxzg7045`. Upload authorship cannot establish line-level algorithm ownership. No original history, repository visibility, files, or Git configuration were changed.

The portfolio starts fresh Git history. The old `.git` directory, duplicate toolbox trees, and raw private evidence are excluded.

## Attribution boundaries

| Layer | Credit and evidence | Claim boundary |
| --- | --- | --- |
| Toolbox, supervised/classical baselines, common loaders | rPPG-Toolbox contributors and cited architecture authors | Not Logan's original toolbox or algorithms. |
| Multi-task waveform objectives and noise-aware processing | ISVC paper authors: Fangshi Zhou, Tianming Zhao, Adam Holsinger, Zhongmei Yao | Published work predates Logan's 2026 integration. |
| Four-model extension and UBFC → PURE comparison | Collaborative team; shared README headed “Fangshi extended Models & Ablation Study” | Retained as previously documented team results. |
| ROI method research | Matthew Dowell; source comment “MediaPipe ROI extractor (Matthew)” and Logan's confirmation | Logan implemented/integrated the researched approach; no independent invention claim. |
| Motion extraction / 3DDFA workflow | Patrick Conley; decks and code helper comments | Logan integrated features and batch support, not the underlying tracker. |
| Configuration, integration, debugging, experiments | Logan's ten progress decks, cross-checked against source | Only supported completed items use first person. |
| Feedback and advising | Fangshi's feedback in slides and loader comments; Dr. Tianming Zhao confirmed as advisor by Logan | Feedback-derived changes are credited as collaborative. |

## Contribution evidence register

Ten decks from the private Logan presentation ZIP were inspected as evidence. Slide text, all available notes XML, embedded screenshots, diagrams, plots and captions were reviewed; notes contained only slide-number placeholders, not substantive additional authorship evidence. Embedded screenshots corroborated code interfaces, preprocessing logs and extraction shapes. One ROI-validation image contained an identifiable face and remains private. Some text placeholders for waveform pictures coexist with embedded plots; no missing placeholder is treated as a completed result.

The decks are not bundled. Identifiers below let Logan locate the original evidence without exposing internal reports or machine paths. Slide numbers are the underlying presentation slide numbers, not alphabetical filename ordering.

| ID | Private deck / slides | Evidence and cross-check |
| --- | --- | --- |
| E01 | rPPG MultiTask GitHub, 2–12 | Pipeline audit, hardcoded ROI/channels/spatial dimensions; artifact weighting and distance interfaces proposed. Does not establish those proposals were implemented. |
| E02 | Logan GitHub Updating, 3–8 | ROI extraction/crop modularization and six-channel input audit; screenshots show interface/factory. Source: `dataset/extraction/roi_extractor.py`, BaseLoader and config. |
| E03 | MediaPipe ROI Integration & Pipeline Verification, 2–8 | Glabella integration, lazy/process-local construction, output `(T,72,72,3)`, preprocessing/train sanity screenshots. Notes MediaPipe 0.10.5 and pre-existing routing/output issues. |
| E04 | rPPG_Weekly_Progress, 2–6 | 100% fallback diagnosis, static-image/refinement/padding fixes and crop visualization. Reports qualitative 5-epoch results; v1 MediaPipe still flat. Region-wide study and 60-epoch comparison remain next steps here. |
| E05 | rPPG_Motion_Update, 2–5 | Patrick's MediaPipe features; PURE batch runner, motion chunk loading/saving, synthetic tests, worker quota reduction. Trainer delivery/routing is explicitly pending in this earlier deck; later E06 and code confirm it. |
| E06 | rPPG_Motion_runs, 2–6 | Shared trainer integration, motion gating, configured temporal depth, identity initialization and per-clip normalization; 10-epoch PURE comparison. Code contains these changes. Global scaling and matched Haar+motion comparison are future work. |
| E07 | rPPG_Update_Daily_Dataset, 2–5 | Daily/Aya integration, six interface/config issues, 241/108 chunks and flat 5-epoch baseline; lower LR, 28 ROI/SQI and broader conditions proposed. Aya loader and main routing support the integration; exact six-fix history is not separately preserved. |
| E08 | rPPG_Daily2, 1–8 | Daily run progression; temporal depth, weighted-loss removal, chunk-wise wavelets, lower SNR weighting, gradient/NaN diagnostics. Source supports these changes, not a conclusion that every run improved every metric. |
| E09 | rppg_daily_motion, 3–11 | Patrick's Daily tracker outputs, Aya loading, session-path and timestamp alignment work under Fangshi's feedback; Run 5 regression, channel diagnosis and normalization limitation. Slide says channels 0/1 corrected; current source still computes eye gradients. Fixed/global scaling remains pending. |
| E10 | rPPG_Update, 2–8 | 60-epoch collapse investigation, Agg plotting, DTW scalar distance/flattening and environment fixes. Slide's 7.43/11.30 bpm figures belong to a 1-epoch sanity run, not the 60-epoch run; 10× PPG weighting is a proposed fix here. |

The PURE reports sometimes say “53 subjects”; the screenshots show 41 training + 12 test files/sessions. PURE identifiers encode subject-trial recordings, so these counts must not be interpreted as 53 unique people. The batch runner targets up to 60 subject-trial folders. The Daily motion deck similarly labels 21 train + 9 test “subjects,” but the configured split has seven train and three test subject IDs across three distances: these are 30 sessions.

## Preparation changes

- Preserved one selected source tree, inherited model/trainer interfaces, metric implementations, published-method references and legitimate notices.
- Replaced machine-specific data/cache/example paths with relative paths. Where dataset identity was apparent from the path, it was preserved. No split, sampling rate, learning rate, loss coefficient, ROI choice or active model was changed.
- Added the existing TS_CAN_Adam trainer to train/test routing and restored explicit unknown-model errors in `main.py`. This makes a documented variant reachable; it does not alter its model computation.
- Added imported utilities missing from the historical requirements. Replaced the environment-deleting setup script with creation of a separate environment.
- Repaired the truncated `DATA_FO` line in the inherited SCAMPS-to-UBFC EfficientPhys inference YAML to `DATA_FORMAT: NDCHW`, matching the sibling EfficientPhys inference recipes. No model or signal calculation changed.
- Restructured documentation and retained approved original figures. No models were retrained and no experiment values were regenerated.

The exact historical upstream revision remains unidentified. Logan confirmed on October 21, 2026 that Patrick Conley and the relevant team/rightsholders approved public redistribution of the team additions, including Patrick-derived motion helpers; this does not assign a new license or transfer copyright. See [third-party notices](../THIRD_PARTY_NOTICES.md).

Hashes, remaining reproducibility gaps and actual checks are recorded separately. Proposed improvements were not retroactively implemented to make the research look complete.
