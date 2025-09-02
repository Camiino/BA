# Motion Capture Processing Pipeline (Bachelor Thesis)

This repository contains a reproducible, script-based pipeline to clean, interpolate, segment, average, and visualize optical motion-capture trajectories exported as CSV files. The code converts raw exports into consistent, smoothed trajectories; derives scalar velocity/acceleration; and produces per-subject, per-experiment, and demographic-cluster summaries and plots.

The pipeline is intentionally stepwise: each stage writes its results to an `Exports/` subfolder so you can inspect intermediate outputs, tune parameters, and re-run selectively.


## Overview

- Input: Raw CSV exports (with a `TRAJECTORIES` section) from your capture software, placed under `Exports/Daten_Raw/`.
- Processing: Formatting -> parsing/cleaning -> interpolation -> trimming -> within-subject averaging -> across-subjects averaging -> smoothing -> plots.
- Output: Final, cleaned averages in `Exports/Final_Cleaned/` and scalar kinematic plots in `Exports/Plots/`. Additional cluster averages and plots are written to `Exports/Clustered/`.


## Environment

- Python 3.10+
- Packages: `numpy`, `pandas`, `scipy`, `matplotlib`

Install:

```
pip install numpy pandas scipy matplotlib
```

All scripts are plain Python entry points. Run each with `python script.py` from the repo root.


## Repository Structure

Key scripts and their outputs (in order of execution):

- `Step1_csv_formatter.py` -> `Exports/Daten_Raw_Formatted/`
- `Step2_data_clean_recursive.py` -> `Exports/Daten_Raw_Clean/`
- `Step2-5_references.py` -> `Exports/Reference/`
- `Step3_interpolate.py` -> `Exports/Daten_Raw_Interpolated/`
- `Step4_trim.py` -> `Exports/Daten_Trimmed/`
- `Step5_average_1M.py` -> `Exports/Daten_Averaged_1M/`
- `Step6_average_by_experiment_1M.py` -> `Exports/Final_Averages_1M/`
- `Step7_clean_final_data.py` -> `Exports/Final_Cleaned/`
- `Step8_plots.py` -> `Exports/Plots/`
- `Step9_Segmentation.py` -> `Exports/Clustered/...`
- `visualize_marker_trajectories.py` -> ad-hoc 3D marker animation from any CSV.

Notes:
- `data_clean_v1.py` exists but is not part of the official pipeline described below.
- Zipped raw acquisitions under `Exports/Probant*/...` are not consumed directly by these steps; the pipeline expects CSVs under `Exports/Daten_Raw/`.


## Data Model and Naming

- CSV delimiter is normalized to `;` throughout the pipeline.
- A standard trajectory table has columns: `Frame;1_X;1_Y;1_Z;2_X;...;5_Z` (marker IDs 1-5, axes X/Y/Z). Some experiments only use a subset.
- Filenames include both proband and experiment (e.g., `Probant3_kreis_....csv`).
  - Proband extraction uses `probant\d+`.
  - Experiment detection relies on substrings: `kreis`, `ptp`, `ptp2`, `ptp3`, `zikzak`, `sequentiell`, `praezision`, `greifen`, `gewicht`.


## Processing Pipeline (Step-by-Step)

1) Step 1 — CSV Formatting
- File: `Step1_csv_formatter.py`
- Purpose: Converts heterogeneous raw CSVs (`,` delimiter, `cp1252` encoding) into consistent `;`-separated files with rectangular shape under `Exports/Daten_Raw_Formatted/`.
- Key details:
  - Finds the `TRAJECTORIES` line and pads all following rows to equal length.
  - Preserves header lines and converts `,` to `;`.

2) Step 2 — Parse and Clean Formatted CSVs
- File: `Step2_data_clean_recursive.py`
- Purpose: Parses the formatted files into numeric tables; extracts `Frame` and per-marker axis columns; resolves duplicate/extra markers.
- Highlights:
  - Detects marker numbers from header tokens like `*1`, and maps coordinates from the following row.
  - Numeric parsing is robust to odd decimal patterns.
  - Heuristic `determine_base_markers()` decides which marker IDs must be present per experiment group, and copies data from “extra” markers into missing base markers when appropriate.
  - Outputs to `Exports/Daten_Raw_Clean/`.

3) Step 2.5 — Select Reference Trials
- File: `Step2-5_references.py`
- Purpose: For experiments `gewicht`, `greifen`, `praezision`, chooses the files with the most valid rows for markers 4 and 5.
- Output: `Exports/Reference/marker{4,5}_{experiment}.csv` — used as motion templates to fill long leading/trailing gaps in Step 3.

4) Step 3 — Interpolate and Fill Gaps
- File: `Step3_interpolate.py`
- Purpose: Resolves missing values inside and at the ends of sequences.
- Methods:
  - Internal gaps: linear interpolation (`limit_area='inside'`).
  - Leading/trailing gaps: propagate deltas from neighbor markers (e.g., for marker 2, use 1 and 3) to continue trajectories forward/backward.
  - If available, reference trials (Step 2.5) drive long end fills for markers 4 and 5 by replaying reference deltas.
- Output: `Exports/Daten_Raw_Interpolated/` and a report of remaining NaNs.

5) Step 4 — Trim Passive Tails
- File: `Step4_trim.py`
- Purpose: Removes the late, inactive tail of each sequence.
- Method: Computes frame-to-frame Euclidean displacement across all markers; uses a rolling mean and truncates after the last index above `MOVEMENT_THRESHOLD`.
- Params: `WINDOW_SIZE`, `MOVEMENT_THRESHOLD`, `FPS`.
- Output: `Exports/Daten_Trimmed/`.

6) Step 5 — Average Trials Per Proband and Experiment
- File: `Step5_average_1M.py`
- Purpose: For each `(probant, experiment)` group, resamples each trimmed trial to the average frame count, then takes the mean and std across trials.
- Notes:
  - Optional `NORMALIZE_START` subtracts the initial position per column to align starts.
  - For `gewicht/greifen/praezision`, only marker `#3` is kept.
- Output: `Exports/Daten_Averaged_1M/*_{experiment}_mean.csv` and `*_std.csv`.

7) Step 6 — Average Across Probands (by Experiment)
- File: `Step6_average_by_experiment_1M.py`
- Purpose: Aggregates all `*_mean.csv` for a given experiment across probands, resamples to the average length, and outputs a single across-subject mean.
- Output: `Exports/Final_Averages_1M/{experiment}_final_mean.csv` with `Frame` normalized to `0..100` percent.

8) Step 7 — Final Smoothing
- File: `Step7_clean_final_data.py`
- Purpose: Applies Savitzky-Golay smoothing to the across-subject means, column-wise.
- Params: `WINDOW_SIZE=11` (odd), `POLY_ORDER=2`.
- Output: `Exports/Final_Cleaned/*_final_clean.csv`.

9) Step 8 — Scalar Velocity/Acceleration Plots
- File: `Step8_plots.py`
- Purpose: From the cleaned positions, computes scalar speed (|v|) and acceleration (|a|) per marker; trims to the main movement; and plots.
- Methods and rationale:
  - Robust pre-filtering with a Hampel-like outlier guard.
  - Position smoothing via Savitzky-Golay (`window=21`, `poly=3`).
  - Derivatives via Savitzky-Golay (`deriv=1/2`) using the average `Frame` delta; avoids amplifying noise compared to naive finite differences while preserving peak timing.
  - Light magnitude smoothing: velocity window 9, acceleration window 15 (`poly=2`).
  - Main-segment detection: hysteresis around the global |v| peak; defaults `enter=5%`, `exit=3%`, with a minimum duration of `10%` of the series; time is re-normalized to `0..100%` for the plot.
  - Optional dominant-peak emphasis for |v|: adaptive SG smoothing with `PEAK_V_WINDOW_FRAC` (e.g., 0.15–0.33) to suppress small ripples while preserving the main lobe height.
- Output: `Exports/Plots/*_velocity_acc_scalar.png`.

10) Step 9 — Clustered Averages and Plots
- File: `Step9_Segmentation.py`
- Purpose: Builds demographic clusters (by gender, age-within-gender, and age only) using a `probanten_dict` mapping from probant number to metadata; averages within each cluster and experiment and plots scalar kinematics.
- Output: `Exports/Clustered/<Mode>/<Cluster>/<Experiment>/{mean.csv, velocity_acc_scalar.png}`.

11) 3D Trajectory Visualization (ad-hoc)
- File: `visualize_marker_trajectories.py`
- Purpose: Loads any CSV with `m_X/m_Y/m_Z` columns, removes invalid rows, and animates markers in 3D using `matplotlib.animation`.
- Usage: `python visualize_marker_trajectories.py <path/to/csv>`


## Running the Pipeline

Assuming your raw CSVs are under `Exports/Daten_Raw/`:

1. Format raw CSVs
```
python Step1_csv_formatter.py
```
2. Parse/clean
```
python Step2_data_clean_recursive.py
```
3. Pick references for fallback (optional, but used by Step 3)
```
python Step2-5_references.py
```
4. Interpolate + end fills
```
python Step3_interpolate.py
```
5. Trim passive tails
```
python Step4_trim.py
```
6. Average per probant/experiment
```
python Step5_average_1M.py
```
7. Average across probands (by experiment)
```
python Step6_average_by_experiment_1M.py
```
8. Final smoothing
```
python Step7_clean_final_data.py
```
9. Plots (|v|, |a|)
```
python Step8_plots.py
```
10. Clustered summaries (optional)
```
python Step9_Segmentation.py
```

You can re-run any single step after tuning its parameters; downstream steps will read the updated outputs.


## Methods and Rationale (short)

- Smoothing: Savitzky-Golay preserves peak timing and amplitude better than simple moving averages or unfitted Butterworth in the time domain, while avoiding phase lag. Positions are smoothed prior to differentiation to prevent noise amplification.
- Differentiation: SG derivatives (`deriv=1/2`) directly estimate v/a from local polynomial fits — more robust to noise than finite differences.
- Interpolation & Gap Filling: Internal gaps are linearly interpolated; long leading/trailing gaps are filled using neighbor markers’ motion deltas or reference trials.
- Trimming: Rolling displacement detects the end of relevant motion and removes quiescent tails.
- Temporal Alignment: Resampling to the average length followed by a normalized percent timeline allows averaging across trials and probands without explicit time-warping.
- Segmentation for plots: Hysteresis around the global speed peak isolates the movement phase of interest and removes pre/post noise.


## Important Parameters to Tune

- Step 3 (`Step3_interpolate.py`): neighbor relations in `SPECIAL_NEIGHBORS`, choice of `FALLBACK_MARKERS` and references.
- Step 4 (`Step4_trim.py`): `WINDOW_SIZE`, `MOVEMENT_THRESHOLD`.
- Step 5 (`Step5_average_1M.py`): `NORMALIZE_START`, `SINGLE_MARKER_EXPERIMENTS`.
- Step 7 (`Step7_clean_final_data.py`): `WINDOW_SIZE`, `POLY_ORDER`.
- Step 8 (`Step8_plots.py`):
  - Hampel filter: `HAMPEL_WINDOW`, `HAMPEL_SIGMAS`.
  - SG position/derivative windows: `SG_WINDOW_POS`, `SG_WINDOW_DER`, `SG_POLY_POS`.
  - Segment detection: `_find_main_segment(enter_frac=0.05, exit_frac=0.03, min_len_frac=0.1)`.
  - Dominant-peak emphasis: `EMPHASIZE_VELOCITY`, `PEAK_V_WINDOW_FRAC` (e.g., 0.15–0.33), `PEAK_POLY`.


## Expected Outputs

- `Exports/Daten_Raw_Formatted/`: formatted raw data (`;` delimited).
- `Exports/Daten_Raw_Clean/`: parsed, numeric tables.
- `Exports/Reference/`: best reference trials for markers 4/5 (select experiments).
- `Exports/Daten_Raw_Interpolated/`: interpolated and end-filled data.
- `Exports/Daten_Trimmed/`: truncated sequences (movement only).
- `Exports/Daten_Averaged_1M/`: per-proband means and stds.
- `Exports/Final_Averages_1M/`: experiment-level means (Frame in `%`).
- `Exports/Final_Cleaned/`: smoothed final CSVs.
- `Exports/Plots/`: scalar |v|/|a| plots per experiment.
- `Exports/Clustered/...`: cluster-level means and plots.


## Re-running on New Data

1. Place new raw CSVs under `Exports/Daten_Raw/`.
2. Run steps 1->9 as above.
3. Inspect outputs at each stage, adjust thresholds/windows if needed.

Tip: If file naming differs, update the experiment/probant detection rules in `Step5_average_1M.py` and `Step6_average_by_experiment_1M.py`.


## Troubleshooting

- Encoding issues (e.g., strange characters in names): Ensure consistent `utf-8` for later steps; Step 1/2 read `cp1252` when parsing raw exports.
- Missing columns: Verify that cleaned CSVs contain `Frame` and `m_X/m_Y/m_Z` columns for expected markers.
- Too many small peaks in |v|: Increase `PEAK_V_WINDOW_FRAC` (e.g., 0.20–0.33) or `PEAK_V_WINDOW`, or tighten Step 7 smoothing.
- Segment too long/short: Adjust `enter_frac/exit_frac/min_len_frac` in `_find_main_segment`.
- Remaining NaNs after Step 3: Check reference files and neighbor relationships; review the console "Remaining NaNs" report.


## Acknowledgements

This code follows established biomechanical signal-processing practice:
- Smooth positions first, then differentiate (Savitzky-Golay) to preserve peak timing and reduce noise amplification.
- Use robust guards (Hampel) to suppress spikes before smoothing.
- Normalize and resample for averaging across trials and participants.

If you have questions or want to automate the whole chain, consider adding a simple `run_all.py` that calls each step in order with your preferred parameters.

