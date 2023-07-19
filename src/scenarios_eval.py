from copy import deepcopy
from multiprocessing.pool import ThreadPool
import os
import sys

import pandas as pd
from oemof.solph import Model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

from src.district import Scenarios, calculate_kpis, create_energy_system, post_process_results, solve_model
from src.prepare_district_data import prepare_district_dataframe
from src.common import RESULTS_ITERATIONS_DIR, sizing

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
sns.set(font="arial")

# In this example, I will map as a grid the two indicators (cost, co2) based only on the sizing of the RE facilities


def evaluate_pv_wind_sizes(n_days=30, start_day=90, grid_resolution=5, folder = RESULTS_ITERATIONS_DIR):
    """Evaluate PV and wind system sizes based on cost and CO2 emissions indicators.

    Args:
        n_days (int, optional): Number of days to simulate. Defaults to 30.
        start_day (int, optional): Starting day of the simulation. Defaults to 90.
        grid_resolution (int, optional): Grid resolution for PV and wind sizes. Defaults to 5.

    Returns:
        str: The filename of the CSV file containing the evaluation results.
    """
    year = 2020
    days = n_days
    boundary_data = prepare_district_dataframe(year=year, days=days, start_day=start_day)
    list_pv = range(0, 200 + grid_resolution, grid_resolution)
    list_wind = range(0, 200 + grid_resolution, grid_resolution)

    def evaluate_es_reduced(params):
        try:
            es_size = deepcopy(sizing)
            es_size["PV"] = params[0]  # kW
            es_size["WT"] = params[1]  # kW
            scenario = Scenarios.PV_WIND
            data = deepcopy(boundary_data)
            energy_system = create_energy_system(
                boundary_data=data, scenario=scenario, sizing=es_size)
            model = Model(energy_system)
            solved_energy_system = solve_model(model)
            results_sim = post_process_results(solved_energy_system)
            kpis = calculate_kpis(
                results_sim, data, es_size, scenario=scenario)
            cost = kpis["total_system_costs"]
            co2 = kpis["total_co2"]
        except:
            cost = np.nan
            co2 = np.nan

        return cost, co2

    results = {}
    results["PV"] = []
    results["WIND"] = []
    results["TotalCost"] = []
    results["TotalCO2"] = []

    pool = ThreadPool(
        min(len(list_pv) * len(list_wind), 16)
    )
    params_ = []
    for x in list_pv:
        for y in list_wind:
            params_.append([[x, y]])
            results["PV"].append(x)
            results["WIND"].append(y)
    F = pool.starmap(evaluate_es_reduced, params_)
    f1_ = [F[i][0] for i in range(len(F))]
    f2_ = [F[i][1] for i in range(len(F))]

    results["TotalCost"] = f1_
    results["TotalCO2"] = f2_
    pool.close()

    df = pd.DataFrame.from_dict(results)
    results_filename = f"iteration_results_{n_days}d_{grid_resolution}kW.csv"
    df.to_csv(os.path.join(folder, results_filename))
    return results_filename


def evaluate_plot_iterations(results_filename="iteration_results_7d_5kW.csv", folder = RESULTS_ITERATIONS_DIR):
    """Evaluate and plot the results of different PV and wind system sizes.

    Args:
        results_filename (str, optional): The filename of the CSV file containing the evaluation results.
            Defaults to "iteration_results_7d_5kW.csv".
    """
    tail = results_filename.split(".")[0]
    tail = tail.split("_")[-2] + "_" + tail.split("_")[-1]
    data = pd.read_csv(os.path.join(folder, results_filename))
    fig, ax = plt.subplots(figsize=(6, 4))
    data["Total RE installed [kW]"] = data["PV"] + data["WIND"]
    sns.scatterplot(data=data, x="TotalCost", y="TotalCO2", ax=ax, hue="Total RE installed [kW]")
    ax.set_ylabel("Total Emissions [kg CO2]")
    ax.set_xlabel("Total Operation Cost [EUR]")
    ax.set_title("Operational KPIs", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"solutions_{tail}.png"), dpi=600)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    titles = ["Total Cost", "Total Emissions"]

    cols = ["TotalCost", "TotalCO2"]
    for i in range(2):
        ax = axs[i]
        x_min = min(data["PV"])
        y_min = min(data["WIND"])
        x_max = max(data["PV"])
        y_max = max(data["WIND"])

        ax.set_xlim(left=x_min, right=x_max)
        ax.set_ylim(bottom=y_min, top=y_max)
        x = np.sort(np.unique(data["PV"]))
        y = np.sort(np.unique(data["WIND"]))

        n = len(x)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros((n, n))
        col = cols[i]
        title = titles[i]
        for i in range(n):
            for j in range(n):
                pv = float(x[i])
                wind = float(y[j])
                point = data[(data["PV"] == pv) & (data["WIND"] == wind)][col]
                point = list(point)
                Z[i, j] = point[0]

        c = ax.pcolor(X, Y, Z, cmap="GnBu")
        ax.set_ylabel("WT Site [kW]")
        ax.set_xlabel("PV Size [kW]")
        ax.set_title(title, fontweight="bold")

        fig.colorbar(c, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"technologies_{tail}.png"), dpi=600)


def evaluate_plot_results(n_days=7, grid_resolution=40, folder = RESULTS_ITERATIONS_DIR):
    """Evaluate and plot the results of PV and wind system sizes.

    Args:
        n_days (int, optional): Number of days to simulate. Defaults to 7.
        grid_resolution (int, optional): Grid resolution for PV and wind sizes. Defaults to 40.
    """
    results_filename = evaluate_pv_wind_sizes(n_days=n_days, grid_resolution=grid_resolution, folder=folder)
    evaluate_plot_iterations(results_filename, folder=folder)


if __name__ == "__main__":
    evaluate_plot_results(n_days=7, grid_resolution=20)
