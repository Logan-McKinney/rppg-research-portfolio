# Attribution and distribution notices

## Upstream toolbox

This project derives from [ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox). The complete Responsible Artificial Intelligence Source Code License (version 1.1, November 20, 2022) is included in [LICENSE](LICENSE), retrieved from upstream LICENSE blob `67f9f16ce37c2fd7a0c09a1ea308dd14279c79a7` on October 21, 2026. Use and redistribution of upstream-derived work are subject to that license, including the restrictions in Section 3.2 and the downstream obligations stated there. No MIT relicensing is asserted.

The research snapshot did not contain a root license file; the published upstream notice has been restored. The exact historical toolbox revision remains unidentified. On October 21, 2026, Logan confirmed that Patrick Conley and the relevant team/rightsholders approved public redistribution of the collaborative additions, including the Patrick-derived motion helper implementations. This records redistribution permission, not a transfer of copyright or a new blanket license. Upstream-derived work remains subject to the included Responsible AI license; no additional license is assigned to independently owned contributions.

Upstream toolbox authors include Xin Liu, Girish Narayanswamy, Akshay Paruchuri, Xiaoyu Zhang, Jiankai Tang, Yuzhe Zhang, Yuntao Wang, Soumyadip Sengupta, Shwetak Patel, and Daniel McDuff. Copyright remains with the respective authors and contributors. Existing notices in copied files have been retained.

## OpenCV cascade

`dataset/haarcascade_frontalface_default.xml` is retained byte for byte, including its Intel License Agreement, Copyright (C) 2000 Intel Corporation, redistribution conditions and disclaimer. The embedded notice credits Rainer Lienhart as the detector creator. This is a licensed detector asset, not a participant image or learned research checkpoint.

## Team and published research

The two ISVC 2024 papers and their authors are cited in the README. The four-model comparison and original figures come from the shared research README and Images folder, not from newly performed portfolio experiments. Logan confirmed redistribution permission for the architecture diagram and waveform plots on October 21, 2026. Individual figure creators remain attributed collectively to the research team pending more specific records.

Matthew Dowell researched the ROI approach; Logan McKinney implemented/integrated it and documented debugging. Patrick Conley contributed motion tracking and 3DDFA-related work; Logan integrated tracker outputs. Fangshi Zhou provided research direction and feedback. Dr. Tianming Zhao advised Logan. These credits do not transfer ownership of published algorithms to Logan.

## External tracker ecosystem

[MediaPipe](https://github.com/google-ai-edge/mediapipe) and [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) remain external dependencies/projects. No standalone 3DDFA source, pretrained weights, CropSense implementation, or Face Landmarker task file is included. The included PURE batch runner explicitly derives feature-extraction helpers from Patrick's script; approved public redistribution of those helpers is recorded above. References to a Daily tracker using SCRFD/InsightFace come from the private progress evidence, not a claim that those components are bundled or share a single license.

## Data and private evidence

Dataset owners govern access to PURE, UBFC-rPPG, SCAMPS, MMPD, COHFACE and the team's Daily/Aya data. This repository grants no dataset access or redistribution permission. No participant recordings, individual measurements, cached motion exports, private progress decks or model weights are included. The approved team plots contain no visible subject identifiers; permission and visual review do not establish the original run-to-subject mapping.
