# fshd.py
# BME 2315 Module 2: FSHD MRI fat-fraction analysis
#
# This file holds the functions that load the MRI data, check it, calculate
# fat fraction, and summarize it. The code follows the class example
# notebook (FSHD_TA_MRI_example_notebook.ipynb), moved into functions so the
# notebook and main.py can both use it.

from pathlib import Path
import numpy as np
import pandas as pd

# Folder that holds slice_info.csv and all of the .npy image files
DATA_DIR = (
    Path(__file__).resolve().parent
    / "FSHD_MRI_data"
    / "FSHD_TA_MRI_data_0020_TP1_TP2"
)

# Pixels with (water + fat) above this are real tissue.
# Background outside the leg is mostly 1-3, tissue is in the hundreds.
TISSUE_THRESHOLD = 20


def load_manifest():
    # PSEUDOCODE:
    #   read slice_info.csv into a pandas table
    #   (one row per sample: which visit, where along the muscle, and the
    #   names of its water, fat, and mask files)
    manifest = pd.read_csv(DATA_DIR / "slice_info.csv")
    return manifest


def load_sample(row):
    # PSEUDOCODE:
    #   use the file names in this row of the manifest to load the
    #   water image, fat image, and TA mask
    #   turn the mask into True (TA) / False (not TA)
    water = np.load(DATA_DIR / row["water_file"])
    fat = np.load(DATA_DIR / row["fat_file"])
    mask = np.load(DATA_DIR / row["mask_file"]).astype(bool)
    return water, fat, mask


def check_sample(row):
    # PSEUDOCODE:
    #   load the water, fat, and mask files for this row
    #   check that all three images are the same size
    #   check that the mask only has 0s and 1s
    #   check that there are no missing (NaN) values
    #   check that no signal value is negative
    #   check that every TA pixel has some signal (so we never divide by 0)
    #   check that the mask isn't empty
    #   return all the answers as True/False
    water = np.load(DATA_DIR / row["water_file"])
    fat = np.load(DATA_DIR / row["fat_file"])
    mask_values = np.load(DATA_DIR / row["mask_file"])   # raw 0s and 1s
    mask = mask_values.astype(bool)
    denominator = water + fat

    checks = {
        "same size": water.shape == fat.shape == mask.shape,
        "mask is only 0s and 1s": np.array_equal(np.unique(mask_values), [0, 1]),
        "no missing values": not (np.isnan(water).any() or np.isnan(fat).any()),
        "no negative signal": water.min() >= 0 and fat.min() >= 0,
        "every TA pixel has signal": np.all(denominator[mask] > 0),
        "mask is not empty": mask.sum() > 0,
    }
    return checks


def calculate_fat_fraction_full(water, fat):
    # PSEUDOCODE:
    #   denominator = water + fat
    #   fat fraction = 100 * fat / denominator, only where denominator > 0
    #   pixels with no signal are set to 0 (so we never divide by 0)
    denominator = water + fat
    fat_fraction = np.divide(
        fat,
        denominator,
        out=np.zeros_like(fat, dtype=float),
        where=denominator > 0
    ) * 100
    return fat_fraction


def calculate_fat_fraction(water, fat, mask):
    """Return a masked fat-fraction image in percent."""
    # PSEUDOCODE:
    #   calculate fat fraction for every pixel (same as above)
    #   set every pixel outside the TA mask to NaN so it's left out of
    #   the mean, median, and histogram
    result = calculate_fat_fraction_full(water, fat)
    result[~mask] = np.nan
    return result


def region_values(water, fat, mask):
    # PSEUDOCODE:
    #   return the fat fraction values for three regions:
    #       whole image = any pixel with signal (includes background noise)
    #       leg tissue  = pixels with real tissue signal (above threshold)
    #       TA only     = pixels inside the TA mask
    fat_fraction_full = calculate_fat_fraction_full(water, fat)
    fat_fraction_masked = calculate_fat_fraction(water, fat, mask)
    denominator = water + fat

    whole_image_values = fat_fraction_full[denominator > 0]
    leg_values = fat_fraction_full[denominator > TISSUE_THRESHOLD]
    muscle_values = fat_fraction_masked[~np.isnan(fat_fraction_masked)]
    return whole_image_values, leg_values, muscle_values


def summarize_visit(manifest, timepoint):
    # PSEUDOCODE:
    #   keep only the rows for this visit (TP1 or TP2)
    #   for each row (each location along the muscle):
    #       load water, fat, mask
    #       calculate the masked fat fraction
    #       record the mean, median, SD, and number of TA pixels
    #   return the results as a table (one row per location)
    visit_results = []
    visit_manifest = manifest[manifest["timepoint"] == timepoint]

    for _, row in visit_manifest.iterrows():
        water, fat, mask = load_sample(row)
        ff = calculate_fat_fraction(water, fat, mask)

        visit_results.append({
            "subject": row["subject"],
            "timepoint": row["timepoint"],
            "slice_id": row["slice_id"],
            "percent_along_muscle": row["percent_along_muscle"],
            "distance_from_muscle_start_mm": row["distance_from_muscle_start_mm"],
            "mean_fat_fraction_percent": np.nanmean(ff),
            "median_fat_fraction_percent": np.nanmedian(ff),
            "std_fat_fraction_percent": np.nanstd(ff),
            "n_muscle_pixels": int(mask.sum()),
        })

    return pd.DataFrame(visit_results)


def compare_visits(profiles_df, value="mean_fat_fraction_percent"):
    # PSEUDOCODE:
    #   make a table with one row per location and one column per visit
    #   add a column for the change: TP2 - TP1
    comparison = profiles_df.pivot(
        index="percent_along_muscle",
        columns="timepoint",
        values=value
    )
    comparison["change_TP2_minus_TP1"] = comparison["TP2"] - comparison["TP1"]
    return comparison