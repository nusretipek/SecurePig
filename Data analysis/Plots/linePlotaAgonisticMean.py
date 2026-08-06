import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Revised agonistic timeline plot:
#
# - Shows treatment means only
# - Adds error bars at each observation interval
# - Uses proportional x-axis spacing based on actual experimental days
# - Shows interval labels below the x-axis
# - Removes replicate lines and horizontal grid lines
#
# Unit:
#   total bouts per treatment replicate per day
#
# Counting logic:
#   Each CSV row is one behavioural event.
#   Initiator + Receiver gives two animal participations per event.
#   Dividing by 2 converts participations back to event/bout counts.
#   Dividing by days gives total bouts per replicate per day.
#
# The treatment mean is then calculated across the four replicate rounds.
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot agonistic behaviour treatment averages with error bars."
    )

    parser.add_argument(
        "--missing-animals",
        default="../missingAnimals.json",
        help="Path to missingAnimals.json. Default: ../missingAnimals.json",
    )

    parser.add_argument(
        "--output-dir",
        default="../output",
        help="Base output directory containing behaviour folders. Default: ../output",
    )

    parser.add_argument(
        "--output",
        default="../output/agonisticMean.pdf",
        help="Output PDF path.",
    )

    parser.add_argument(
        "--error",
        choices=["sem", "sd", "ci95"],
        default="sem",
        help="Error bars across replicate totals: sem, sd, or ci95. Default: sem",
    )

    return parser.parse_args()


def get_error(values, error_type="sem"):
    """Calculate SEM, SD, or 95% CI across replicate values."""
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

    raise ValueError("error_type must be one of: sem, sd, ci95")


def get_behaviour_replicate_values(behaviour, missing_animals_all, output_dir):
    """
    Returns replicate-level values for one behaviour.

    Output:
        d["Control"][expID] = [D0-D2, D2-D7, D7-D9, D9-D17]
        d["Pheromone"][expID] = [D0-D2, D2-D7, D7-D9, D9-D17]

    Each value is:
        total bouts per replicate per day
    """

    d = {
        "Pheromone": {2: [], 4: [], 5: [], 8: []},
        "Control": {3: [], 6: [], 7: [], 9: []},
    }

    period_files = ["0_2", "2_7", "7_9", "9_17"]
    period_days = np.array([2, 5, 2, 8], dtype=float)

    control_experiments = {
        3: "Experiment3",
        6: "Experiment6",
        7: "Experiment7",
        9: "Experiment9",
    }

    pheromone_experiments = {
        2: "Experiment2",
        4: "Experiment4",
        5: "Experiment5",
        8: "ExperimentRemux8",
    }

    for exp_id, exp_name in control_experiments.items():
        for idx, period_file in enumerate(period_files):
            file_path = f"{output_dir}/{behaviour}/{exp_name}/{period_file}.csv"
            data = pd.read_csv(file_path)

            all_values = pd.concat([data["Initiator"], data["Receiver"]])

            total_bouts = len(all_values) / 2
            bouts_per_replicate_per_day = total_bouts / period_days[idx]

            d["Control"][exp_id].append(bouts_per_replicate_per_day)

    for exp_id, exp_name in pheromone_experiments.items():
        for idx, period_file in enumerate(period_files):
            file_path = f"{output_dir}/{behaviour}/{exp_name}/{period_file}.csv"
            data = pd.read_csv(file_path)

            all_values = pd.concat([data["Initiator"], data["Receiver"]])

            total_bouts = len(all_values) / 2
            bouts_per_replicate_per_day = total_bouts / period_days[idx]

            d["Pheromone"][exp_id].append(bouts_per_replicate_per_day)

    return d


def summarise_treatment_means(d, error_type):
    """
    Convert replicate dictionaries into treatment means and errors.

    SEM/SD/CI are calculated across the four replicate-level treatment totals.
    """

    control_rounds = [3, 6, 7, 9]
    pheromone_rounds = [2, 4, 5, 8]

    control_array = np.array(
        [d["Control"][r] for r in control_rounds],
        dtype=float,
    )

    pheromone_array = np.array(
        [d["Pheromone"][r] for r in pheromone_rounds],
        dtype=float,
    )

    control_mean = control_array.mean(axis=0)
    pheromone_mean = pheromone_array.mean(axis=0)

    control_error = np.array([
        get_error(control_array[:, i], error_type)
        for i in range(control_array.shape[1])
    ])

    pheromone_error = np.array([
        get_error(pheromone_array[:, i], error_type)
        for i in range(pheromone_array.shape[1])
    ])

    return control_array, pheromone_array, control_mean, pheromone_mean, control_error, pheromone_error


def main():
    args = parse_args()

    control_color = "#2a8f7b"
    pheromone_color = "#8b2a86"

    error_label = {
        "sem": "SEM",
        "sd": "SD",
        "ci95": "95% CI",
    }[args.error]

    interval_starts = np.array([0, 2, 7, 9], dtype=float)
    interval_ends = np.array([2, 7, 9, 17], dtype=float)

    x_positions = (interval_starts + interval_ends) / 2

    boundary_ticks = np.array([0, 2, 7, 9, 17], dtype=float)
    boundary_labels = ["D0", "D2", "D7", "D9", "D17"]

    period_labels = ["D0-D2", "D2-D7", "D7-D9", "D9-D17"]

    # Kept for compatibility with earlier workflow, although not needed
    # for treatment-level totals.
    if os.path.exists(args.missing_animals):
        with open(args.missing_animals, "r") as file:
            _ = json.load(file)

    data_fight = get_behaviour_replicate_values(
        behaviour="circling",
        missing_animals_all=None,
        output_dir=args.output_dir,
    )

    data_flee = get_behaviour_replicate_values(
        behaviour="fleeing",
        missing_animals_all=None,
        output_dir=args.output_dir,
    )

    data_disp = get_behaviour_replicate_values(
        behaviour="replacement",
        missing_animals_all=None,
        output_dir=args.output_dir,
    )

    fight_summary = summarise_treatment_means(data_fight, args.error)
    flee_summary = summarise_treatment_means(data_flee, args.error)
    disp_summary = summarise_treatment_means(data_disp, args.error)

    plot_specs = [
        ("Fighting", fight_summary),
        ("Fleeing", flee_summary),
        ("Displacement", disp_summary),
    ]

    print(f"\nError bars: {error_label}")
    print("Values are treatment means in total bouts per replicate per day.\n")

    for behaviour_name, summary in plot_specs:
        (
            control_array,
            pheromone_array,
            control_mean,
            pheromone_mean,
            control_error,
            pheromone_error,
        ) = summary

        print(f"\n{behaviour_name}")
        print("Control")
        for label, mean, err in zip(period_labels, control_mean, control_error):
            print(f"  {label}: mean={mean:.3f}, {error_label}={err:.3f}")

        print("Pheromone")
        for label, mean, err in zip(period_labels, pheromone_mean, pheromone_error):
            print(f"  {label}: mean={mean:.3f}, {error_label}={err:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True)

    for ax, (behaviour_name, summary) in zip(axes, plot_specs):
        (
            control_array,
            pheromone_array,
            control_mean,
            pheromone_mean,
            control_error,
            pheromone_error,
        ) = summary

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

        ax.set_title(behaviour_name, fontsize=15, weight='bold')
        ax.set_ylabel(f"Behaviour Bouts (per day) ± {error_label}", fontsize=13)

        ax.set_xlim(0, 17)
        ax.set_xticks(boundary_ticks)
        ax.set_xticklabels(boundary_labels, fontsize=12)
        #ax.set_xlabel("Experimental day", fontsize=13)

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
                weight='bold'
                
            )

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=12)

        max_y = np.nanmax([
            np.nanmax(control_mean + np.nan_to_num(control_error)),
            np.nanmax(pheromone_mean + np.nan_to_num(pheromone_error)),
        ])

        ax.set_ylim(0, max_y * 1.25 if max_y > 0 else 1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=13,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()