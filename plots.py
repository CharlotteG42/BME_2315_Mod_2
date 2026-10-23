# plots.py
# BME 2315 Module 2: FSHD MRI fat-fraction and interpolation analysis
#
# This file holds the functions that make our 9 figures.
#   Figures 1-4: MRI Data Analysis Methodology (Figures 1-3 use the same
#                plotting code as the class example notebook)
#   Figure 5:    Interpolation along the muscle (location)
#   Figures 6-9: Interpolation over time (follow the class interpolation example)
# Each function makes one figure. If you give it a file name (save_as), it
# saves the figure there; if not, it shows the figure (this is what the
# notebook does).

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from fshd import load_sample, calculate_fat_fraction, calculate_fat_fraction_full
from interpolation import MEASURES, interpolate_subject

# Same colors for each visit (MRI) and each time point (interpolation)
VISIT_COLORS = {"TP1": "#2a78d6", "TP2": "#eb6834"}   # blue, orange
TIME_COLORS = {"T1": "#2a78d6", "T2": "#eb6834"}      # blue, orange


def show_or_save(save_as):
    # PSEUDOCODE:
    #   if no file name was given, show the figure (notebook)
    #   otherwise save it to that file and close it (main.py)
    if save_as is None:
        plt.show()
    else:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Figure 1: water, fat, and fat fraction images for one sample
# ---------------------------------------------------------------------------

def plot_example_slice(water, fat, mask, title, save_as=None):
    # PSEUDOCODE:
    #   calculate fat fraction for the whole image and for the TA only
    #   find the bounding box of the TA mask (smallest/largest row and column)
    #   crop the TA-only fat fraction map to that box
    #   panel 1: water image with the TA outlined in red
    #   panel 2: fat image with the TA outlined in red
    #   panel 3: fat fraction of the whole image (0-100%), TA outlined in cyan
    #   panel 4: zoomed-in fat fraction of the TA only (0-100%)
    fat_fraction_full = calculate_fat_fraction_full(water, fat)
    fat_fraction_masked = calculate_fat_fraction(water, fat, mask)

    # Find the bounding box of the TA mask
    y_pixels, x_pixels = np.where(mask)
    y0, y1 = y_pixels.min(), y_pixels.max() + 1
    x0, x1 = x_pixels.min(), x_pixels.max() + 1

    # Crop the TA-specific fat-fraction map
    fat_fraction_TA = fat_fraction_masked[y0:y1, x0:x1]

    # 2 x 2 grid of panels (axes.flatten() turns it into a list of 4)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    # Water image with TA mask outline
    axes[0].imshow(water, cmap="gray")
    axes[0].contour(mask, levels=[0.5], colors="red", linewidths=2)
    axes[0].set_title("A. Water image + TA mask")

    # Fat image with TA mask outline
    axes[1].imshow(fat, cmap="gray")
    axes[1].contour(mask, levels=[0.5], colors="red", linewidths=2)
    axes[1].set_title("B. Fat image + TA mask")

    # Fat fraction for the whole supplied image region
    axes[2].imshow(fat_fraction_full, cmap="plasma", vmin=0, vmax=100)
    axes[2].contour(mask, levels=[0.5], colors="cyan", linewidths=2)
    axes[2].set_title("C. Fat fraction: whole region")

    # Zoomed-in fat fraction for the TA only
    ta_image = axes[3].imshow(fat_fraction_TA, cmap="plasma", vmin=0, vmax=100)
    axes[3].set_title("D. TA fat fraction (zoomed in)")

    for ax in axes:
        ax.axis("off")

    # Both fat-fraction plots use the same 0-100% scale
    fig.colorbar(ta_image, ax=axes[2:], fraction=0.046, pad=0.04,
                 label="Fat fraction (%)")
    fig.suptitle(title)
    show_or_save(save_as)


# ---------------------------------------------------------------------------
# Figure 2: fat fraction distribution for one sample
# ---------------------------------------------------------------------------

def plot_histogram(whole_image_values, leg_values, muscle_values, title,
                   save_as=None):
    # PSEUDOCODE:
    #   plot a histogram of fat fraction for the whole image, leg tissue,
    #   and TA (30 bins)
    #   use density so the groups can be compared even though they have
    #   very different numbers of pixels
    #   draw lines at the TA mean and TA median
    plt.figure(figsize=(8, 4.5))
    plt.hist(whole_image_values, bins=30, density=True, alpha=0.45,
             color="gray", label="Whole supplied region")
    plt.hist(leg_values, bins=30, density=True, histtype="step",
             color="black", linewidth=1.5, label="Leg tissue only")
    plt.hist(muscle_values, bins=30, density=True, alpha=0.65,
             color=VISIT_COLORS["TP1"], label="TA muscle only")

    plt.axvline(np.mean(muscle_values), color="black", linestyle="-",
                label=f"TA mean = {np.mean(muscle_values):.1f}%")
    plt.axvline(np.median(muscle_values), color="black", linestyle="--",
                label=f"TA median = {np.median(muscle_values):.1f}%")

    plt.xlabel("Fat fraction (%)")
    plt.ylabel("Density (fraction of pixels per %)")
    plt.title(title)
    plt.legend()
    show_or_save(save_as)


# ---------------------------------------------------------------------------
# Figure 3: fat fraction along the muscle at each visit
# ---------------------------------------------------------------------------

def plot_profile(profiles_df, save_as=None):
    # PSEUDOCODE:
    #   for each visit (TP1, TP2):
    #       sort the rows by distance from the start of the muscle
    #       plot mean fat fraction vs distance (mm), error bars = SD
    #       plot median fat fraction vs distance as a dashed line
    #   (distance in mm goes on the x-axis; the class example labeled its
    #   x-axis in mm but plotted percent, so we fixed that here)
    plt.figure(figsize=(9, 5))
    for timepoint, group in profiles_df.groupby("timepoint"):
        group = group.sort_values("distance_from_muscle_start_mm")
        plt.errorbar(
            group["distance_from_muscle_start_mm"],
            group["mean_fat_fraction_percent"],
            yerr=group["std_fat_fraction_percent"],
            marker="o",
            capsize=3,
            color=VISIT_COLORS[timepoint],
            label=f"{timepoint} mean (bars = SD)"
        )
        plt.plot(
            group["distance_from_muscle_start_mm"],
            group["median_fat_fraction_percent"],
            marker="o",
            linestyle="--",
            markerfacecolor="white",
            color=VISIT_COLORS[timepoint],
            label=f"{timepoint} median"
        )
    plt.ylim(0, 105)   # fat fraction can't go below 0%
    plt.xlabel("Distance from muscle start (mm)")
    plt.ylabel("TA fat fraction (%)")
    plt.title("Fat fraction along the tibialis anterior, TP1 vs. TP2")
    plt.legend(title="Visit", ncol=2)
    plt.grid(True, alpha=0.3)
    show_or_save(save_as)


# ---------------------------------------------------------------------------
# Figure 4: grid of TA fat fraction maps, TP1 vs TP2
# ---------------------------------------------------------------------------

def plot_map_grid(manifest, save_as=None):
    # PSEUDOCODE:
    #   make a 4 x 4 grid of small images
    #       rows 1-2: TP1 and TP2 at 10%, 20%, 30%, 40%
    #       rows 3-4: TP1 and TP2 at 50%, 60%, 70%, 80%
    #   for each location:
    #       load the TP1 and TP2 samples and calculate the masked fat fraction
    #       find one bounding box that fits the TA at BOTH visits
    #       (so both visits are zoomed the same amount)
    #       show the TP1 map, and the TP2 map right below it
    # layout="constrained" spaces the panels so labels don't overlap
    fig, axes = plt.subplots(4, 4, figsize=(11, 12), layout="constrained")
    percents = sorted(manifest["percent_along_muscle"].unique())

    for i, percent in enumerate(percents):
        rows = manifest[manifest["percent_along_muscle"] == percent]
        tp1_row = rows[rows["timepoint"] == "TP1"].iloc[0]
        tp2_row = rows[rows["timepoint"] == "TP2"].iloc[0]

        # Load both visits and calculate their masked fat fraction
        water1, fat1, mask1 = load_sample(tp1_row)
        water2, fat2, mask2 = load_sample(tp2_row)
        ff1 = calculate_fat_fraction(water1, fat1, mask1)
        ff2 = calculate_fat_fraction(water2, fat2, mask2)

        # Bounding box that covers the TA at both visits
        # (mask1 | mask2 means "TA at TP1 or TA at TP2")
        y_pixels, x_pixels = np.where(mask1 | mask2)
        y0, y1 = y_pixels.min(), y_pixels.max() + 1
        x0, x1 = x_pixels.min(), x_pixels.max() + 1

        # Where this location goes in the grid:
        # 10-40% use rows 0 (TP1) and 1 (TP2); 50-80% use rows 2 and 3
        if i < 4:
            tp1_ax = axes[0, i]
            tp2_ax = axes[1, i]
        else:
            tp1_ax = axes[2, i - 4]
            tp2_ax = axes[3, i - 4]

        image = tp1_ax.imshow(ff1[y0:y1, x0:x1], cmap="plasma", vmin=0, vmax=100)
        tp2_ax.imshow(ff2[y0:y1, x0:x1], cmap="plasma", vmin=0, vmax=100)

        tp1_ax.set_title(f"{percent}% ({tp1_row['distance_from_muscle_start_mm']:.0f} mm)")
        tp1_ax.set_xlabel(f"mean {np.nanmean(ff1):.0f}%")
        tp2_ax.set_xlabel(f"mean {np.nanmean(ff2):.0f}%")

    # Labels and clean axes
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    for row_number, label in enumerate(["TP1", "TP2", "TP1", "TP2"]):
        axes[row_number, 0].set_ylabel(label, fontsize=13, fontweight="bold")

    # One color bar for the whole grid
    fig.colorbar(image, ax=axes, shrink=0.6, label="Fat fraction (%)")
    fig.suptitle("TA-only fat fraction maps along the muscle\n"
                 "(titles: % of muscle length, TP1 distance from start)")
    show_or_save(save_as)


# ===========================================================================
# INTERPOLATION ANALYSIS FIGURES
# ===========================================================================

# ---------------------------------------------------------------------------
# Figure 5: interpolation along the muscle (location)
# ---------------------------------------------------------------------------

def plot_location_interpolation(profiles_df, target_percent, estimates,
                                save_as=None):
    # PSEUDOCODE:
    #   for each visit (TP1, TP2):
    #       plot mean fat fraction at the 8 sampled locations, joined by
    #       straight lines (the straight lines ARE the linear interpolation)
    #       mark the estimate at the target location with a diamond
    #   draw a dashed vertical line at the target location
    plt.figure(figsize=(9, 5))
    for timepoint, group in profiles_df.groupby("timepoint"):
        group = group.sort_values("percent_along_muscle")
        plt.plot(group["percent_along_muscle"],
                 group["mean_fat_fraction_percent"],
                 marker="o", color=VISIT_COLORS[timepoint],
                 label=f"{timepoint} measured")
        plt.scatter([target_percent], [estimates[timepoint]], marker="D",
                    s=110, color=VISIT_COLORS[timepoint], edgecolor="black",
                    zorder=4, label=f"{timepoint} estimate = "
                                    f"{estimates[timepoint]:.1f}%")
    plt.axvline(target_percent, color="gray", linestyle="--")
    plt.ylim(0, 105)   # fat fraction can't go below 0%
    plt.xlabel("Location along the muscle (% of muscle length)")
    plt.ylabel("Mean TA fat fraction (%)")
    plt.title(f"Linear interpolation at {target_percent}% along the TA")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    show_or_save(save_as)


# ---------------------------------------------------------------------------
# Figure 6: interpolation for one example subject
# ---------------------------------------------------------------------------

def plot_interpolation_example(df, subject_id, measures, T1, T2, save_as=None):
    # PSEUDOCODE:
    #   find this subject's row in the dataset
    #   make one panel per measure (2 x 2 grid)
    #   for each measure:
    #       plot the 3 measured visits, joined by straight lines
    #           (the straight lines ARE the linear interpolation)
    #       mark the interpolated estimates at T1 and T2 with diamonds
    #       draw dashed vertical lines at T1 and T2
    row = df[df["subject_id"] == subject_id].iloc[0]
    times = [row["lag_months_baseline"], row["lag_months_y1"], row["lag_months_y2"]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes = axes.flatten()

    for i, measure in enumerate(measures):
        info = MEASURES[measure]
        values = [row[c] for c in info["columns"]]
        value_T1, value_T2 = interpolate_subject(row, info["columns"], T1, T2)
        ax = axes[i]

        # Measured visits joined by straight lines
        ax.plot(times, values, color="gray", linestyle="--", zorder=1)
        ax.scatter(times, values, color="black", s=60, zorder=3,
                   label="Measured visits")

        # Interpolated estimates at T1 and T2
        ax.scatter([T1], [value_T1], marker="D", s=90, zorder=4,
                   color=TIME_COLORS["T1"], label=f"Estimate at T1 = {T1:.1f} mo")
        ax.scatter([T2], [value_T2], marker="D", s=90, zorder=4,
                   color=TIME_COLORS["T2"], label=f"Estimate at T2 = {T2:.1f} mo")
        ax.axvline(T1, color=TIME_COLORS["T1"], linestyle=":", alpha=0.6)
        ax.axvline(T2, color=TIME_COLORS["T2"], linestyle=":", alpha=0.6)

        ax.set_title(info["label"])
        ax.set_xlabel("Months since baseline")
        ax.set_ylabel(f"{info['label']} ({info['units']})")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=9)
    fig.suptitle(f"Linear interpolation for subject {subject_id}")
    show_or_save(save_as)


# ---------------------------------------------------------------------------
# Figure 7: every subject over time (whole cohort)
# ---------------------------------------------------------------------------

def plot_cohort_trajectories(df, results, measures, T1, T2, save_as=None):
    # PSEUDOCODE:
    #   make one panel per measure (2 x 2 grid)
    #   for each measure:
    #       draw one thin gray line per subject through their measured visits
    #       draw one thick colored line through the group average at
    #           baseline, T1, and T2 (subjects with all three values)
    #       draw dashed vertical lines at T1 and T2
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes = axes.flatten()

    for i, measure in enumerate(measures):
        info = MEASURES[measure]
        ax = axes[i]

        # One thin gray line per subject
        for _, row in df.iterrows():
            times = [row["lag_months_baseline"], row["lag_months_y1"],
                     row["lag_months_y2"]]
            values = [row[c] for c in info["columns"]]
            ax.plot(times, values, color="gray", alpha=0.35, linewidth=1)

        # Group average at baseline, T1, and T2
        measure_rows = results[results["measure"] == measure]
        complete = measure_rows.dropna(subset=["baseline", "value_T1", "value_T2"])
        averages = [complete["baseline"].mean(), complete["value_T1"].mean(),
                    complete["value_T2"].mean()]
        ax.plot([0, T1, T2], averages, color="#eb6834", linewidth=3, marker="o",
                label=f"Average at baseline, T1, T2 (n = {len(complete)})")

        ax.axvline(T1, color="black", linestyle=":", alpha=0.5)
        ax.axvline(T2, color="black", linestyle=":", alpha=0.5)
        ax.set_title(info["label"])
        ax.set_xlabel("Months since baseline")
        ax.set_ylabel(f"{info['label']} ({info['units']})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Every subject's measured visits (gray) and the group average")
    show_or_save(save_as)


# ---------------------------------------------------------------------------
# Figure 8: sensitivity to change (SRM) for every measure
# ---------------------------------------------------------------------------

def plot_srm(summary, save_as=None):
    # PSEUDOCODE:
    #   sort the measures by their SRM at T2 (largest at the top)
    #   draw two horizontal bars per measure: SRM at T1 and at T2
    #   shade the region from -0.2 to 0.2 ("trivial" change)
    #   positive SRM = the measure got worse on average
    summary = summary.sort_values("SRM_T2")
    positions = np.arange(len(summary))   # 0, 1, 2, ... one per measure
    bar_height = 0.38

    plt.figure(figsize=(9, 6.5))
    plt.axvspan(-0.2, 0.2, color="gray", alpha=0.15,
                label="Trivial change (|SRM| < 0.2)")
    plt.barh(positions - bar_height / 2, summary["SRM_T1"], height=bar_height,
             color=TIME_COLORS["T1"], label="Baseline to T1 (~11 months)")
    plt.barh(positions + bar_height / 2, summary["SRM_T2"], height=bar_height,
             color=TIME_COLORS["T2"], label="Baseline to T2 (~22 months)")
    plt.axvline(0, color="black", linewidth=1)

    plt.yticks(positions, summary["label"])
    plt.xlabel("Standardized response mean (mean change / SD of change)\n"
               "positive = got worse")
    plt.title("Sensitivity to change of each measure")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    show_or_save(save_as)


# ---------------------------------------------------------------------------
# Figure 9: baseline fat fraction vs change in fat fraction
# ---------------------------------------------------------------------------

def plot_baseline_vs_change(result, threshold=20, save_as=None):
    # PSEUDOCODE:
    #   scatter plot: baseline fat fraction (x) vs change by T2 (y)
    #   fit a straight regression line through the points
    #   print Pearson r, Spearman rho, their p-values, and n on the graph
    #   draw a dashed line at the 20% threshold and a line at zero change
    baseline = result["baseline"]
    change = result["change"]

    # Straight-line fit (slope and intercept)
    fit = stats.linregress(baseline, change)
    line_x = np.array([baseline.min(), baseline.max()])
    line_y = fit.intercept + fit.slope * line_x

    plt.figure(figsize=(8, 5.5))
    plt.scatter(baseline, change, color="#2a78d6", s=60, zorder=3)
    plt.plot(line_x, line_y, color="black", linewidth=2, label="Linear regression")
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(threshold, color="#eb6834", linestyle="--",
                label=f"{threshold}% baseline fat fraction")

    # Print the test results in the top left corner of the graph
    plt.text(0.03, 0.97,
             f"Pearson r = {result['pearson_r']:.2f}, p = {result['pearson_p']:.3f}\n"
             f"Spearman rho = {result['spearman_rho']:.2f}, "
             f"p = {result['spearman_p']:.3f}\n"
             f"n = {result['n']}",
             transform=plt.gca().transAxes, ha="left", va="top",
             bbox=dict(facecolor="white", edgecolor="lightgray"))

    plt.title("Baseline TA fat fraction vs. change by T2 (both legs)")
    plt.xlabel("Baseline fat fraction (%)")
    plt.ylabel("Change in fat fraction by T2 (percentage points)")
    plt.legend(loc="lower center")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    show_or_save(save_as)