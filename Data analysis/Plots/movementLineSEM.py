import os
import ast
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Figure 4 revised for reviewer comment:
#
# Main version:
# - Shows overall treatment means only
# - Adds error bars around means
# - Removes individual replicate lines by default
# - Removes horizontal grid lines
# - Uses a proportional x-axis based on actual experimental days
#
# Optional:
# - Use --replicates to add individual replicate lines with low alpha
#   and the same line styles/legend structure as the old figure.
#
# Important:
# Each y-value represents movement during an interval, not on a single
# exact day. Therefore, values are plotted at the interval midpoints,
# but the x-axis uses true day boundaries: D0, D2, D7, D9, D17.
#
# Movement is normalized as:
#   total movement / number of animals / number of days / 80
#
# Therefore y-axis = movement per sow per day during each interval.
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot treatment mean movement per sow per day with errors."
    )

    parser.add_argument(
        "--missing-animals",
        default="../missingAnimals.json",
        help="Path to missingAnimals.json. Default: ../missingAnimals.json",
    )

    parser.add_argument(
        "--movement-dir",
        default="../output/movement",
        help="Base directory containing movement output folders. Default: ../output/movement",
    )

    parser.add_argument(
        "--output",
        default="../output/movement/movement_mean_per_sow_per_day_line.pdf",
        help="Output PDF path.",
    )

    parser.add_argument(
        "--error",
        choices=["sem", "sd", "ci95"],
        default="sem",
        help="Error bars to plot: sem, sd, or ci95. Default: sem",
    )

    parser.add_argument(
        "--replicates",
        action="store_true",
        help="Add individual replicate lines with low alpha and old line styles.",
    )

    return parser.parse_args()


def get_error(values, error_type="sem"):
    """Return error-bar value for a 1D numeric array."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n <= 1:
        return np.nan

    sd = np.std(x, ddof=1)

    if error_type == "sd":
        return sd
    if error_type == "sem":
        return sd / np.sqrt(n)
    if error_type == "ci95":
        return 1.96 * sd / np.sqrt(n)

    raise ValueError("error_type must be one of: 'sem', 'sd', or 'ci95'")


def bold_math_text(s):
    """Bold label for matplotlib legend entries."""
    return r"$\bf{" + s + "}$"


def main():
    args = parse_args()

    control_color = "#2a8f7b"
    pheromone_color = "#8b2a86"

    line_styles = ["-", "--", "-.", ":"]

    data = {
        "Pheromone": {2: [], 4: [], 5: [], 8: []},
        "Control": {3: [], 6: [], 7: [], 9: []},
    }

    replicate_label_map = {
        "Control": {3: "C1", 6: "C2", 7: "C3", 9: "C4"},
        "Pheromone": {2: "P1", 4: "P2", 5: "P3", 8: "P4"},
    }

    period_files = ["0_2", "2_7", "7_9", "9_17"]
    period_labels = ["D0-D2", "D2-D7", "D7-D9", "D9-D17"]

    interval_starts = np.array([0, 2, 7, 9], dtype=float)
    interval_ends = np.array([2, 7, 9, 17], dtype=float)
    period_durations = interval_ends - interval_starts

    x_positions = (interval_starts + interval_ends) / 2

    boundary_ticks = np.array([0, 2, 7, 9, 17], dtype=float)
    boundary_labels = ["D0", "D2", "D7", "D9", "D17"]

    with open(args.missing_animals, "r") as file:
        missingAnimals = json.load(file)

    for expID in range(2, 10):

        if expID == 8:
            prefix = "ExperimentRemux"
        else:
            prefix = "Experiment"

        missingAnimalExp = missingAnimals[str(expID)]
        animalCount = 21 - len(missingAnimalExp)

        for tJ in period_files:
            movement_file = f"{args.movement_dir}/{prefix}{expID}/{tJ}.txt"

            with open(movement_file, "r") as f:
                content = f.read()
                dataDict = ast.literal_eval(content)

            movement_per_sow = sum(dataDict.values()) / animalCount

            if expID in [2, 4, 5, 8]:
                data["Pheromone"][expID].append(movement_per_sow)
            else:
                data["Control"][expID].append(movement_per_sow)

    control_rounds = sorted(data["Control"].keys())
    pheromone_rounds = sorted(data["Pheromone"].keys())

    control_array = (
        np.array([data["Control"][r] for r in control_rounds], dtype=float)
        / period_durations
        / 80
    )

    pheromone_array = (
        np.array([data["Pheromone"][r] for r in pheromone_rounds], dtype=float)
        / period_durations
        / 80
    )
    
    print("\nIndividual replicate values: movement per sow per day")
    print("Rows are replicate rounds; columns are periods.\n")
    
    print("Control")
    for label, row in zip(["C1", "C2", "C3", "C4"], control_array):
        print(
            f"{label}: "
            + ", ".join(
                f"{period}={value:.3f}"
                for period, value in zip(period_labels, row)
            )
        )
    
    print("\nPheromone")
    for label, row in zip(["P1", "P2", "P3", "P4"], pheromone_array):
        print(
            f"{label}: "
            + ", ".join(
                f"{period}={value:.3f}"
                for period, value in zip(period_labels, row)
            )
        )
    control_mean = np.mean(control_array, axis=0)
    pheromone_mean = np.mean(pheromone_array, axis=0)

    control_error = np.array([
        get_error(control_array[:, i], args.error)
        for i in range(control_array.shape[1])
    ])

    pheromone_error = np.array([
        get_error(pheromone_array[:, i], args.error)
        for i in range(pheromone_array.shape[1])
    ])

    error_label = {
        "sem": "SEM",
        "sd": "SD",
        "ci95": "95% CI",
    }[args.error]

    print("\nMovement values plotted: treatment mean per sow per day during each interval")
    print(f"Error bars: {error_label}")
    print(f"Replicates shown: {args.replicates}")

    print("\nControl")
    for label, mean, err in zip(period_labels, control_mean, control_error):
        print(f"{label}: mean={mean:.3f}, {error_label}={err:.3f}")

    print("\nPheromone")
    for label, mean, err in zip(period_labels, pheromone_mean, pheromone_error):
        print(f"{label}: mean={mean:.3f}, {error_label}={err:.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))

    if args.replicates:
        for idx, expID in enumerate(control_rounds):
            ax.plot(
                x_positions,
                control_array[idx, :],
                linestyle=line_styles[idx % len(line_styles)],
                color=control_color,
                alpha=0.25,
                linewidth=2,
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1,
                markeredgecolor=control_color,
                zorder=1,
            )

        for idx, expID in enumerate(pheromone_rounds):
            ax.plot(
                x_positions,
                pheromone_array[idx, :],
                linestyle=line_styles[idx % len(line_styles)],
                color=pheromone_color,
                alpha=0.25,
                linewidth=2,
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1,
                markeredgecolor=pheromone_color,
                zorder=1,
            )

    ax.errorbar(
        x_positions,
        control_mean,
        yerr=control_error,
        color=control_color,
        label="Control",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.8,
        capsize=5,
        capthick=1.4,
        elinewidth=1.4,
        zorder=5,
    )

    ax.errorbar(
        x_positions,
        pheromone_mean,
        yerr=pheromone_error,
        color=pheromone_color,
        label="Pheromone",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.8,
        capsize=5,
        capthick=1.4,
        elinewidth=1.4,
        zorder=5,
    )
    ax.axvspan(0, 2, color="black", alpha=0.1, zorder=0)
    ax.axvspan(7, 9, color="black", alpha=0.1, zorder=0)

    ax.set_xlim(0, 17)
    ax.set_xticks(boundary_ticks)
    ax.set_xticklabels(boundary_labels, fontsize=12)
    #ax.set_xlabel("Experimental day", fontsize=14)

    trans = ax.get_xaxis_transform()
    for x, label in zip(x_positions, period_labels):
        ax.text(
            x,
            -0.06,
            label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=10,
            weight="bold"
        )

    #ax.text(
    #    8.5,
    #    -0.22,
    #    "Observation interval",
    #    transform=trans,
    #    ha="center",
    #    va="top",
    #    fontsize=12,
    #)

    ax.set_ylabel(f"Movement (m per sow per day) ± {error_label}", fontsize=14)

    max_y = np.nanmax([
        np.nanmax(control_mean + np.nan_to_num(control_error)),
        np.nanmax(pheromone_mean + np.nan_to_num(pheromone_error)),
        np.nanmax(control_array) if args.replicates else 0,
        np.nanmax(pheromone_array) if args.replicates else 0,
    ])

    ax.set_ylim(0, max_y * 1.20)

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

    if args.replicates:
        custom_lines = []

        custom_lines.append(
            Line2D(
                [],
                [],
                color="none",
                linestyle="solid",
                label=bold_math_text("Control"),
                linewidth=0,
            )
        )

        for i, expID in enumerate(control_rounds):
            custom_lines.append(
                Line2D(
                    [],
                    [],
                    color=control_color,
                    linestyle=line_styles[i % len(line_styles)],
                    marker="o",
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    markeredgecolor=control_color,
                    linewidth=2,
                    alpha=0.25,
                    label=f"  {replicate_label_map['Control'][expID]}",
                )
            )

        custom_lines.append(
            Line2D(
                [],
                [],
                color="none",
                linestyle="solid",
                label=bold_math_text("Pheromone"),
                linewidth=0,
            )
        )

        for i, expID in enumerate(pheromone_rounds):
            custom_lines.append(
                Line2D(
                    [],
                    [],
                    color=pheromone_color,
                    linestyle=line_styles[i % len(line_styles)],
                    marker="o",
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    markeredgecolor=pheromone_color,
                    linewidth=2,
                    alpha=0.25,
                    label=f"  {replicate_label_map['Pheromone'][expID]}",
                )
            )

        custom_lines.append(
            Line2D(
                [],
                [],
                color="none",
                linestyle="solid",
                label=bold_math_text("Mean"),
                linewidth=0,
            )
        )

        custom_lines.append(
            Line2D(
                [],
                [],
                color=control_color,
                linestyle="solid",
                linewidth=2.8,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.5,
                markeredgecolor=control_color,
                label="  Control",
            )
        )

        custom_lines.append(
            Line2D(
                [],
                [],
                color=pheromone_color,
                linestyle="solid",
                linewidth=2.8,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.5,
                markeredgecolor=pheromone_color,
                label="  Pheromone",
            )
        )

        ax.legend(
            handles=custom_lines,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            fontsize=10,
            handlelength=2.6,
            borderpad=0,
            labelspacing=0.8,
            title=None,
        )

        plt.setp(ax.get_legend().get_texts(), ha="right")

    else:
        ax.legend(
            frameon=False,
            fontsize=12,
            loc="upper right",
        )

    plt.tight_layout(rect=[0, 0.14, 1, 1])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()