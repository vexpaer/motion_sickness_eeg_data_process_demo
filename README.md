# Public processing code and disclosure-reviewed results

This repository contains the public processing and analysis code
for the motion-sickness EEG dataset, together with disclosure-reviewed
aggregate results used in the accompanying manuscript. It is deliberately **not**
a copy of the private processing workspace.

Repository: <https://github.com/vexpaer/motion_sickness_eeg_data_process_demo>

## What is included

```text
github_upload_code/
|-- README.md
|-- config.yaml
|-- requirements.txt
|-- run_pipeline.py              # scan, crop, preprocess, feature extraction
|-- src/eeg_pipeline/             # public processing modules
`-- results/                      # aggregate, disclosure-reviewed CSV outputs
|   |-- data_integrity/
|   |-- signal_quality/
|   |-- ica/
|   |-- features/
|   |-- statistics/
|   `-- models/
```

No BDF, EDF, FIF, NumPy array, feature matrix, window-level signal, or other
waveform-bearing file is included. The controlled dataset is available only
through the Science Data Bank record and its data-access agreement:
<https://doi.org/10.57760/sciencedb.32147>.

## Reproduction boundary

After obtaining authorized data, place the local `EEG_data/` directory next to
this repository, or pass its path with `--data-root`. The public pipeline reads
`data.bdf` and `evt.bdf` from that external directory and writes generated
intermediate files to a local output directory. The repository does not upload
those files.

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py scan
python run_pipeline.py crop
python run_pipeline.py preprocess
python run_pipeline.py features
```

The processing order is marker-based lap segmentation, 1--40 Hz filtering,
robust bad-channel detection and montage interpolation, average A1/A2
rereferencing, resampling to 250 Hz, per-lap Infomax ICA, and six-window
channel-level feature extraction. The exact thresholds are recorded in
`config.yaml` and the feature definitions are provided in
`results/features/feature_dictionary.csv`.

## Public result files

The `results/` directory contains aggregate summaries only. It includes data
completeness and signal-quality summaries, ICA audit summaries, the complete
feature dictionary, mixed-model, Kruskal--Wallis, permutation and band-power summaries, and the
model metric/calibration tables reported in the manuscript. Individual-lap
signals, per-window feature values, out-of-fold predictions, and fold-level
files are intentionally excluded.

Manuscript figures, table images, access statements, DUA files and confidential
reviewer instructions are maintained in the journal submission package rather
than in this code repository. The confidential reviewer-only access instructions
and private reviewer link are not part of this public repository.

## License and use

Code and aggregate result files are provided for reproducibility of the
published work. Dataset access, human-data handling, and redistribution are
governed by the Science Data Bank record and the DUA, not by this code archive.
