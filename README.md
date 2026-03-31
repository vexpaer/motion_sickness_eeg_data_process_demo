# Motion Sickness EEG Data Process Demo

This folder provides 4 example scripts for EEG/EMG preprocessing and feature extraction.
All examples are configured to use subject folder 25009 under the current working directory.

## 1. Recommended Demo Folder Layout

Run scripts in this project directory, and prepare data like this:

```text
motion_sickness_eeg_data_process_demo/
	25009/
		data.bdf
		evt.bdf
		crop.edf
		... (optional: *crop*.edf, named files for feature extraction)
```

## 2. Dependencies

Required Python packages:

```bash
pip install mne numpy pandas scipy tqdm matplotlib
```

## 3. Example Scripts

### 3.1 preprocess.py

Purpose:
- EEG preprocessing pipeline for one BDF file.
- Steps: montage setup, manual bad-channel interpolation, re-reference (A1/A2), bandpass + notch filtering, ICA artifact removal.

Input:
- ./25009/data.bdf

Output:
- ./25009/data_clean.edf

How to run:

```bash
python preprocess.py
```

Notes:
- This script is interactive (plots will pop up).
- You need to select bad channels and inspect ICA components manually.

### 3.2 crop_by_lap.py

Purpose:
- Segment lap-based static periods from BDF using event markers.
- Splits into 6 segments and exports each as an EDF file.

Input:
- ./25009/data.bdf
- ./25009/evt.bdf

Output:
- ./25009/middle1.edf
- ./25009/middle2.edf
- ...
- ./25009/middle6.edf

How to run:

```bash
python crop_by_lap.py
```

### 3.3 baseline.py

Purpose:
- Perform baseline correction using channel-wise mean values from crop.edf.
- Apply correction to all files matching *crop*.edf in the same folder.

Input:
- Baseline file: ./25009/crop.edf
- Target files: ./25009/*crop*.edf

Output:
- For each target file, export a corrected file with crop replaced by final.

How to run:

```bash
python baseline.py
```

### 3.4 extract_feature.py

Purpose:
- Unified EEG feature extraction script with optional interactive selection.
- Supports 6 feature groups:
	- time (variance, RMS)
	- hjorth
	- psd (delta/theta/alpha/beta)
	- de (differential entropy)
	- plv (phase locking value)
	- wpli (weighted phase lag index)

Default behavior:
- Reads files in ./25009
- Splits each file into 6 segments
- Applies filtering and optional resampling
- Writes one combined CSV

Default output:
- ./features_25009.csv

How to run:

```bash
python extract_feature.py
```

Useful arguments:

```bash
python extract_feature.py --data-dir 25009 --output features_25009.csv --no-interactive
python extract_feature.py --features time,hjorth,psd
python extract_feature.py --channels Fp1,Fp2,C3,C4
```

## 4. Typical Demo Order

1. Run crop_by_lap.py to generate segment files.
2. Run preprocess.py for preprocessing of raw BDF.
3. Run baseline.py for baseline correction (if needed).
4. Run extract_feature.py to export feature table.

## 5. Notes

- File naming rules in extract_feature.py are strict for automatic discovery.
- If a script reports missing channels or files, check the 25009 folder first.
- All paths are relative to current working directory for easy GitHub demonstration.
