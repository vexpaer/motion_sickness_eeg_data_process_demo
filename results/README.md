# Aggregate results

These files are disclosure-reviewed summaries used in the manuscript. They
are not the private workspace outputs. Files containing raw or processed EEG,
window-level feature matrices, out-of-fold predictions, fold assignments, or
participant-level signal traces are intentionally excluded.

| Directory | Contents |
|---|---|
| `data_integrity/` | condition-level session, lap, window, and resting-record counts |
| `signal_quality/` | condition-level amplitude, spectral, and bad-channel summaries |
| `ica/` | condition-level ICA component and pre/post quality audits |
| `features/` | reconstruction-ready feature dictionary and channel-level QC summaries |
| `statistics/` | MISC mixed-model, Kruskal--Wallis and participant-stratified permutation tests, channel-level summary, band-power (PSD) validation, parametric-bootstrap LRT, alternative partitions, and condition/susceptibility summaries |
| `models/` | aggregate benchmark metrics, sensitivity results, label distributions, AUC intervals, and calibration summaries |
