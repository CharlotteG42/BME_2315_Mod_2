# interpolation.py
# BME 2315 Module 2: FSHD interpolation analysis
#
# This file holds the functions for the second half of the project. It only
# DEFINES functions, so running it on its own does nothing; the notebook (or
# main.py) calls them. There are two parts:
#   Part A: interpolating fat fraction at a LOCATION along the TA that wasn't
#           imaged (uses the MRI results from fshd.py)
#   Part B: interpolating every subject in Module2_FSHD_Dataset.csv to two
#           common TIME points and comparing how sensitive each measure is
# The interpolation code follows the class example (Module2_interpolation_example).
#
# Keep this file and Module2_FSHD_Dataset.csv in the same folder as the
# notebook (and main.py).

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats

DATASET_FILE = "Module2_FSHD_Dataset.csv"

# The measures we compare. For each one:
#   "columns" = the baseline, Year 1, and Year 2 columns in the CSV
#   "worse"   = +1 if a HIGHER value means the disease is worse
#               -1 if a LOWER value means worse (strength goes down)
MEASURES = {
    "ff_mean": {"label": "TA fat fraction (both legs)", "units": "%",
                "columns": ["ff_baseline_mean", "ff_y1_mean", "ff_y2_mean"], "worse": 1},
    "ff_R": {"label": "TA fat fraction (right)", "units": "%",
             "columns": ["ff_baseline_R", "ff_y1_R", "ff_y2_R"], "worse": 1},
    "ff_L": {"label": "TA fat fraction (left)", "units": "%",
             "columns": ["ff_baseline_L", "ff_y1_L", "ff_y2_L"], "worse": 1},
    "fvar_R": {"label": "Fat fraction variability (right)", "units": "%",
               "columns": ["fvar_baseline_R", "fvar_y1_R", "fvar_y2_R"], "worse": 1},
    "fvar_L": {"label": "Fat fraction variability (left)", "units": "%",
               "columns": ["fvar_baseline_L", "fvar_y1_L", "fvar_y2_L"], "worse": 1},
    "qmt_R": {"label": "Dorsiflexion strength (right)", "units": "kg",
              "columns": ["dorsiflexion_qmt_R_baseline", "dorsiflexion_qmt_R_y1",
                          "dorsiflexion_qmt_R_y2"], "worse": -1},
    "qmt_L": {"label": "Dorsiflexion strength (left)", "units": "kg",
              "columns": ["dorsiflexion_qmt_L_baseline", "dorsiflexion_qmt_L_y1",
                          "dorsiflexion_qmt_L_y2"], "worse": -1},
    "tug": {"label": "Timed up-and-go", "units": "s",
            "columns": ["tug_time_baseline", "tug_time_y1", "tug_time_y2"], "worse": 1},
    "wr_10": {"label": "10 m walk/run", "units": "s",
              "columns": ["wr_10_baseline", "wr_10_y1", "wr_10_y2"], "worse": 1},
    "ascend_4": {"label": "Stair ascend (4 steps)", "units": "s",
                 "columns": ["ascend_4_baseline", "ascend_4_y1", "ascend_4_y2"], "worse": 1},
    "descend_4": {"label": "Stair descend (4 steps)", "units": "s",
                  "columns": ["descend_4_baseline", "descend_4_y1", "descend_4_y2"], "worse": 1},
    "css": {"label": "Clinical severity score", "units": "points",
            "columns": ["clinical_severity_score_baseline", "clinical_severity_score_y1",
                        "clinical_severity_score_y2"], "worse": 1},
}


# ===========================================================================
# PART A: INTERPOLATING ALONG THE MUSCLE (LOCATION)
# ===========================================================================

def get_visit_profile(profiles_df, timepoint):
    # PSEUDOCODE:
    #   keep the rows for one visit (TP1 or TP2)
    #   sort them by location along the muscle (10% to 80%)
    #   x = percent of muscle length, y = mean TA fat fraction
    visit = profiles_df[profiles_df["timepoint"] == timepoint]
    visit = visit.sort_values("percent_along_muscle")
    x = visit["percent_along_muscle"].to_numpy()
    y = visit["mean_fat_fraction_percent"].to_numpy()
    return x, y


def interpolate_location(profiles_df, timepoint, target_percent):
    # PSEUDOCODE:
    #   get the 8 sampled locations and their mean fat fraction for one visit
    #   check that the target is inside the sampled range (10% to 80%)
    #       (if not, it would be extrapolation, so return NaN)
    #   draw straight lines between the sampled locations (interp1d)
    #   return the estimate at the target location
    x, y = get_visit_profile(profiles_df, timepoint)
    if not x.min() <= target_percent <= x.max():
        return np.nan
    f_linear = interp1d(x, y, kind="linear")
    return float(f_linear(target_percent))


def leave_one_out(profiles_df, timepoint):
    # PSEUDOCODE:
    #   for each inside location (20% to 70%; the ends have only one neighbor):
    #       pretend that location was never imaged
    #       estimate it with a straight line between its two neighbors
    #       error = estimate - real measured value
    #   return one row per location
    x, y = get_visit_profile(profiles_df, timepoint)
    rows = []
    for i in range(1, len(x) - 1):
        f_linear = interp1d([x[i - 1], x[i + 1]], [y[i - 1], y[i + 1]],
                            kind="linear")
        estimate = float(f_linear(x[i]))
        rows.append({"timepoint": timepoint,
                     "percent_along_muscle": x[i],
                     "measured": y[i],
                     "estimate": estimate,
                     "error": estimate - y[i]})
    return pd.DataFrame(rows)


# ===========================================================================
# PART B: INTERPOLATING OVER TIME (34 SUBJECTS)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Loading the data and choosing the common time points
# ---------------------------------------------------------------------------

def load_dataset():
    # PSEUDOCODE:
    #   read Module2_FSHD_Dataset.csv into a pandas table (one row per subject)
    #   for each visit, add a "both legs" fat fraction column:
    #       the average of the right and left TA fat fraction
    df = pd.read_csv(DATASET_FILE)
    for visit in ["baseline", "y1", "y2"]:
        df[f"ff_{visit}_mean"] = (df[f"ff_{visit}_R"] + df[f"ff_{visit}_L"]) / 2
    return df


def get_common_time_points(df):
    # PSEUDOCODE:
    #   T1 = the earliest Year 1 visit of any subject (in months)
    #   T2 = the earliest Year 2 visit of any subject
    #   (every subject's own visits reach at least this far, so we only
    #   ever interpolate, never extrapolate)
    T1 = df["lag_months_y1"].min()
    T2 = df["lag_months_y2"].min()
    return T1, T2


# ---------------------------------------------------------------------------
# 2. Interpolating each subject
# ---------------------------------------------------------------------------

def interpolate_subject(row, columns, T1, T2):
    # PSEUDOCODE:
    #   get this subject's 3 visit times (months) and 3 measured values
    #   estimate at T1: draw a straight line between baseline and Year 1
    #       (only if both of those visits have a value)
    #   estimate at T2: draw straight lines through all 3 visits
    #       (only if all 3 have values and the visit times go in order)
    #   if there isn't enough data, the estimate is NaN (missing)
    times = np.array([row["lag_months_baseline"], row["lag_months_y1"],
                      row["lag_months_y2"]], dtype=float)
    values = np.array([row[columns[0]], row[columns[1]], row[columns[2]]],
                      dtype=float)

    # --- Estimate at T1 (baseline and Year 1) ---
    value_T1 = np.nan
    if not np.isnan(times[:2]).any() and not np.isnan(values[:2]).any():
        f_linear = interp1d(times[:2], values[:2], kind="linear")
        value_T1 = float(f_linear(T1))

    # --- Estimate at T2 (all three visits) ---
    value_T2 = np.nan
    all_present = not (np.isnan(times).any() or np.isnan(values).any())
    if all_present and times[0] < times[1] < times[2]:
        f_linear = interp1d(times, values, kind="linear")
        value_T2 = float(f_linear(T2))

    return value_T1, value_T2


def interpolate_all(df, T1, T2):
    # PSEUDOCODE:
    #   for each subject (each row of the dataset):
    #       for each measure:
    #           interpolate to T1 and T2
    #           change = interpolated value - baseline value
    #   return one table with a row for every subject + measure
    results = []
    for _, row in df.iterrows():
        for measure, info in MEASURES.items():
            baseline = row[info["columns"][0]]
            value_T1, value_T2 = interpolate_subject(row, info["columns"], T1, T2)
            results.append({
                "subject_id": row["subject_id"],
                "measure": measure,
                "baseline": baseline,
                "value_T1": value_T1,
                "value_T2": value_T2,
                "change_T1": value_T1 - baseline,
                "change_T2": value_T2 - baseline,
            })
    return pd.DataFrame(results)


def class_example_estimates(df, T1, T2):
    # Part 6 of the class interpolation example, done for every subject.
    # PSEUDOCODE:
    #   for each subject, using right TA fat fraction (ff_R):
    #       linear estimate at T1 (needs baseline + Year 1)
    #       quadratic estimate at T2 (needs all three visits, in order)
    #       if there isn't enough data, record NaN instead of crashing
    #   return subject ID + both estimates (to save as a .csv)
    columns = MEASURES["ff_R"]["columns"]
    rows = []
    for _, row in df.iterrows():
        times = np.array([row["lag_months_baseline"], row["lag_months_y1"],
                          row["lag_months_y2"]], dtype=float)
        values = np.array([row[c] for c in columns], dtype=float)

        # Linear estimate at T1 (baseline and Year 1)
        estimate_T1 = np.nan
        if not np.isnan(times[:2]).any() and not np.isnan(values[:2]).any():
            f_linear = interp1d(times[:2], values[:2], kind="linear")
            estimate_T1 = float(f_linear(T1))

        # Quadratic estimate at T2 (all three visits)
        estimate_T2 = np.nan
        all_present = not (np.isnan(times).any() or np.isnan(values).any())
        if all_present and times[0] < times[1] < times[2]:
            f_quadratic = interp1d(times, values, kind="quadratic")
            estimate_T2 = float(f_quadratic(T2))

        rows.append({"subject_id": row["subject_id"],
                     "estimate_T1_linear": estimate_T1,
                     "estimate_T2_quadratic": estimate_T2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Comparing how sensitive each measure is to change
# ---------------------------------------------------------------------------

def summarize_sensitivity(results):
    # PSEUDOCODE:
    #   for each measure, at T1 and at T2:
    #       keep the subjects who have both a baseline and an estimate
    #       flip the sign of the change if needed, so positive = got worse
    #       SRM (standardized response mean) = mean change / SD of change
    #       % worse = percent of subjects whose value got worse
    #       paired t-test: is the value at T different from baseline?
    #   return one row per measure
    summary = []
    for measure, info in MEASURES.items():
        measure_rows = results[results["measure"] == measure]
        row = {"measure": measure, "label": info["label"], "units": info["units"]}

        for time_label in ["T1", "T2"]:
            paired = measure_rows.dropna(subset=["baseline", f"value_{time_label}"])
            change = paired[f"change_{time_label}"]
            change_worse = change * info["worse"]      # positive = got worse

            t_stat, p_value = stats.ttest_rel(paired[f"value_{time_label}"],
                                              paired["baseline"])

            row[f"n_{time_label}"] = len(paired)
            row[f"mean_change_{time_label}"] = change.mean()
            row[f"sd_change_{time_label}"] = change.std()
            row[f"SRM_{time_label}"] = change_worse.mean() / change_worse.std()
            row[f"pct_worse_{time_label}"] = 100 * (change_worse > 0).mean()
            row[f"p_{time_label}"] = p_value

        summary.append(row)
    return pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# 4. Primary question: does baseline fat fraction predict change?
# ---------------------------------------------------------------------------

def baseline_vs_change(results, measure="ff_mean", time_label="T2"):
    # PSEUDOCODE:
    #   keep the subjects who have a baseline value and a change at T
    #   Pearson r: is there a straight-line relationship?
    #   Spearman rho: does change tend to go up as baseline goes up
    #       (works for curved relationships and is less affected by outliers)
    #   return the values and both test results
    measure_rows = results[results["measure"] == measure]
    paired = measure_rows.dropna(subset=["baseline", f"change_{time_label}"])
    baseline = paired["baseline"].to_numpy()
    change = paired[f"change_{time_label}"].to_numpy()

    pearson_r, pearson_p = stats.pearsonr(baseline, change)
    spearman_rho, spearman_p = stats.spearmanr(baseline, change)

    return {"baseline": baseline, "change": change, "n": len(baseline),
            "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_rho": spearman_rho, "spearman_p": spearman_p}


def compare_groups(results, measure="ff_mean", time_label="T2", threshold=20):
    # PSEUDOCODE:
    #   split subjects into two groups by baseline fat fraction:
    #       low  = below the threshold (20%)
    #       high = at or above the threshold
    #   Welch's t-test: is the average change different between the groups?
    #   (Welch's version doesn't assume the groups have equal variance,
    #   which matters because the groups are very different sizes)
    measure_rows = results[results["measure"] == measure]
    paired = measure_rows.dropna(subset=["baseline", f"change_{time_label}"])
    low = paired[paired["baseline"] < threshold][f"change_{time_label}"]
    high = paired[paired["baseline"] >= threshold][f"change_{time_label}"]

    t_stat, p_value = stats.ttest_ind(high, low, equal_var=False)
    return {"low": low.to_numpy(), "high": high.to_numpy(),
            "t_stat": t_stat, "p_value": p_value}


# ---------------------------------------------------------------------------
# 5. Verifying the interpolation
# ---------------------------------------------------------------------------

def check_no_extrapolation(df, T1, T2):
    # PSEUDOCODE:
    #   count subjects whose own Year 1 visit is earlier than T1
    #   count subjects whose own Year 2 visit is earlier than T2
    #   (both should be 0; otherwise we'd be extrapolating for them)
    #   also count how many would be extrapolated at exactly 12 and 24 months
    return {
        "before_T1": int((df["lag_months_y1"] < T1).sum()),
        "before_T2": int((df["lag_months_y2"] < T2).sum()),
        "before_12": int((df["lag_months_y1"] < 12).sum()),
        "before_24": int((df["lag_months_y2"] < 24).sum()),
    }


def compare_linear_quadratic(df, T2, measure="ff_mean"):
    # PSEUDOCODE:
    #   for each subject with all three visits:
    #       estimate the value at T2 with a straight line (linear)
    #       estimate it again with a curve through all 3 points (quadratic)
    #       record the difference between the two estimates
    #   (small differences = our choice of linear doesn't change the answer)
    columns = MEASURES[measure]["columns"]
    differences = []
    for _, row in df.iterrows():
        times = np.array([row["lag_months_baseline"], row["lag_months_y1"],
                          row["lag_months_y2"]], dtype=float)
        values = np.array([row[c] for c in columns], dtype=float)
        if np.isnan(times).any() or np.isnan(values).any():
            continue
        if not times[0] < times[1] < times[2]:
            continue
        linear = float(interp1d(times, values, kind="linear")(T2))
        quadratic = float(interp1d(times, values, kind="quadratic")(T2))
        differences.append(quadratic - linear)
    return np.array(differences)