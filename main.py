# main.py
# BME 2315 Module 2: FSHD MRI fat-fraction and interpolation analysis
#
# Runs the whole analysis from start to finish without the notebook.
# Prints the results and saves the 9 figures (figures/ folder) and three
# tables (TA_fat_fraction_summary_TP1_TP2.csv, interpolated_estimates.csv,
# and interpolated_estimates_ff_R.csv).
#
# PSEUDOCODE (the whole pipeline):
#   1. load slice_info.csv (16 rows: 8 locations x 2 visits)
#   2. run the data checks on every sample and stop if any fail
#   3. for one example sample (TP1, 40%):
#          calculate fat fraction = 100 * fat / (water + fat)
#          compare the whole image vs leg tissue vs the TA only
#          make Figure 1 (images) and Figure 2 (histogram)
#   4. summarize every sample at both visits (mean, median, SD, TA pixels)
#   5. compare TP1 vs TP2 at each location
#   6. make Figure 3 (profile) and Figure 4 (map grid)
#   7. save the summary table to a CSV file
#   --- Interpolation analysis ---
#   8. along the muscle: estimate fat fraction at 45% (not imaged) at both
#      visits, check it against real slices, and make Figure 5
#   9. load Module2_FSHD_Dataset.csv and find T1 and T2
#  10. linearly interpolate every subject and measure to T1 and T2, and make
#      the class example table (ff_R: linear at T1, quadratic at T2)
#  11. compare measures with the SRM and paired t-tests
#  12. baseline fat fraction vs change (Pearson, Spearman, Welch's t-test)
#  13. make Figures 6-9 and save the interpolated estimates to CSV files
#
# To run: keep fshd.py, plots.py, interpolation.py, the MRI data folder, and
# Module2_FSHD_Dataset.csv in this same folder, then run "python main.py"
# in the VS Code terminal.

import os
import numpy as np
import pandas as pd
from fshd import *
from interpolation import *
import plots

# Folder for the saved figures (made if it doesn't exist yet)
os.makedirs("figures", exist_ok=True)

# --- 1. Loading the manifest ---
manifest = load_manifest()
print(f"Number of samples = {len(manifest)}")   # should be 16

# --- 2. Checking the data for errors ---
samples_passed = 0
for _, row in manifest.iterrows():
    checks = check_sample(row)
    if all(checks.values()):
        samples_passed += 1
    else:
        print(f"Failed a check: {row['timepoint']} slice {row['slice_id']} -> {checks}")
print(f"Samples that passed every check = {samples_passed} of {len(manifest)}")
if samples_passed != len(manifest):
    raise SystemExit("Fix the data before running the analysis.")

# --- 3. Example sample: TP1 at 40% along the muscle ---
row = manifest[manifest["timepoint"] == "TP1"].iloc[3]   # representative mid-muscle slice
water, fat, mask = load_sample(row)
whole_image_values, leg_values, muscle_values = region_values(water, fat, mask)

print("\nExample sample: TP1, 40% along the muscle")
for name, values in [("Whole supplied region", whole_image_values),
                     ("Leg tissue only", leg_values),
                     ("TA muscle only", muscle_values)]:
    print(f"{name}: pixels = {len(values)}, mean = {np.mean(values):.1f}%, "
          f"median = {np.median(values):.1f}%")

title = "TP1 at 40% along the left TA"
plots.plot_example_slice(water, fat, mask, title,
                         save_as="figures/fig1_example_slice.png")
plots.plot_histogram(whole_image_values, leg_values, muscle_values,
                     "Fat-fraction distributions, " + title,
                     save_as="figures/fig2_histogram.png")

# --- 4. Summarizing every sample at both visits ---
profiles_df = pd.concat(
    [summarize_visit(manifest, "TP1"), summarize_visit(manifest, "TP2")],
    ignore_index=True
)
print("\nSummary of every sample:")
print(profiles_df.round(1).to_string(index=False))

# --- 5. Comparing TP1 vs TP2 at each location ---
print("\nMean fat fraction, TP1 vs TP2:")
print(compare_visits(profiles_df).round(1))
print("\nNumber of TA pixels, TP1 vs TP2:")
print(compare_visits(profiles_df, "n_muscle_pixels"))

# --- 6. Figures along the muscle ---
plots.plot_profile(profiles_df, save_as="figures/fig3_profile.png")
plots.plot_map_grid(manifest, save_as="figures/fig4_map_grid.png")
print("\nSaved Figures 1-4 to the figures folder")

# --- 7. Saving the summary table ---
output_file = "TA_fat_fraction_summary_TP1_TP2.csv"
profiles_df.to_csv(output_file, index=False)
print(f"Saved {output_file}")

# =====================================================================
# INTERPOLATION ANALYSIS
# =====================================================================

# --- 8. Interpolating along the muscle (location) ---
TARGET_PERCENT = 45     # halfway between the 40% and 50% slices (not imaged)
estimates = {}
for timepoint in ["TP1", "TP2"]:
    estimates[timepoint] = interpolate_location(profiles_df, timepoint,
                                                TARGET_PERCENT)
    print(f"\n{timepoint}: estimated fat fraction at {TARGET_PERCENT}% = "
          f"{estimates[timepoint]:.1f}%")
    print(leave_one_out(profiles_df, timepoint).round(1).to_string(index=False))
plots.plot_location_interpolation(profiles_df, TARGET_PERCENT, estimates,
                                  save_as="figures/fig5_location_interpolation.png")

# --- 9. Loading the dataset and choosing T1 and T2 ---
df = load_dataset()
T1, T2 = get_common_time_points(df)
print(f"\nSubjects = {len(df)}, T1 = {T1:.2f} months, T2 = {T2:.2f} months")
print(f"Extrapolation check: {check_no_extrapolation(df, T1, T2)}")

# --- 10. Interpolating every subject and measure ---
results = interpolate_all(df, T1, T2)
class_table = class_example_estimates(df, T1, T2)

# --- 11. Comparing sensitivity to change ---
summary = summarize_sensitivity(results)
print("\nSensitivity to change (positive SRM = got worse):")
print(summary[["label", "n_T2", "SRM_T1", "SRM_T2", "pct_worse_T2", "p_T2"]]
      .sort_values("SRM_T2", ascending=False).round(3).to_string(index=False))

# --- 12. Baseline fat fraction vs change ---
correlation = baseline_vs_change(results, "ff_mean", "T2")
groups = compare_groups(results, "ff_mean", "T2", threshold=20)
print(f"\nBaseline FF vs change: Pearson r = {correlation['pearson_r']:.2f} "
      f"(p = {correlation['pearson_p']:.3f}), Spearman rho = "
      f"{correlation['spearman_rho']:.2f} (p = {correlation['spearman_p']:.3f})")
print(f"Welch's t-test (>= 20% vs < 20%): p = {groups['p_value']:.3f}")

# --- 13. Figures 6-9 and saving the estimates ---
key_measures = ["ff_mean", "qmt_R", "wr_10", "css"]
plots.plot_interpolation_example(df, "13-0004", key_measures, T1, T2,
                                 save_as="figures/fig6_interpolation_example.png")
plots.plot_cohort_trajectories(df, results, key_measures, T1, T2,
                               save_as="figures/fig7_cohort_trajectories.png")
plots.plot_srm(summary, save_as="figures/fig8_srm.png")
plots.plot_baseline_vs_change(correlation, threshold=20,
                              save_as="figures/fig9_baseline_vs_change.png")
print("Saved Figures 5-9 to the figures folder")

results.to_csv("interpolated_estimates.csv", index=False)
class_table.to_csv("interpolated_estimates_ff_R.csv", index=False)
print("Saved interpolated_estimates.csv and interpolated_estimates_ff_R.csv")