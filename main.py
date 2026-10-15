# main.py
# BME 2315 Module 2: FSHD MRI fat-fraction analysis
#
# Runs the whole MRI analysis from start to finish without the notebook.
# Prints the results and saves the 4 figures (figures/ folder) and the
# summary table (TA_fat_fraction_summary_TP1_TP2.csv).
#
# PSEUDOCODE:
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

import os
import numpy as np
import pandas as pd
from fshd import *
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
print("\nSaved 4 figures to the figures folder")

# --- 7. Saving the summary table ---
output_file = "TA_fat_fraction_summary_TP1_TP2.csv"
profiles_df.to_csv(output_file, index=False)
print(f"Saved {output_file}")