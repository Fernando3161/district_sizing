import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib
from copy import deepcopy
import numpy as np

import os
from src.common import RESULTS_YEAR_DIR, RESULTS_OPT_DIR
import statsmodels.api as sm
from numpy import log

sns.set_style("darkgrid")
warnings.filterwarnings("ignore")

def get_pareto_front(df):
    # Add a flag column to identify the Pareto optimal solutions
    df['ParetoOptimal'] = True

    # Iterate over each row to check for dominance
    for i, row in df.iterrows():
        for j, other_row in df.iterrows():
            if i != j:
                if (row[0] > other_row[0] and row[1] > other_row[1]) or (row[0] >= other_row[0] and row[1] > other_row[1]) or (row[0] > other_row[0] and row[1] >= other_row[1]):
                    df.at[i, 'ParetoOptimal'] = False
                    break
    
    # Filter and return only the Pareto optimal solutions
    pareto_front = df[df['ParetoOptimal']]
    
    return pareto_front



def plot_all_solutions(results, output_folder):

    N_GEN =max(results["Gen"])
    fig,ax= plt.subplots(figsize=(6,5))


    for i in range(1,N_GEN+1):
        res_gen = results[results["Gen"]==i]
        res_gen =results[results["total_co2"]!=1e9]
        res_gen["total_system_costs"]*=1/1000
        sns.scatterplot(data=res_gen, x="total_system_costs", y = "total_co2", alpha=(1/((N_GEN-i)+1))**3, edgecolors=None,color="blue")

    ax.set_xlabel("Total System Costs [1000*EUR/year]")
    ax.set_ylabel("Total CO2 Emissions [Ton/year]")
    ax.set_title("Evolution of the Solutions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder,"solutions.png"), dpi=600)

    return None
    # Study Relations Between Indicators


def plot_kpi_pairgrid(results, output_folder):

    optimal = results[results["Gen"] > 0]

    optimal = optimal[optimal["total_co2"] < 1e9]
    optimal = optimal[[
        # 'PV', 'Wind', 'P2H', 'Sto',
        'ex_el_fraction', 'ex_heat_fraction', 'total_co2',
        'total_system_costs']]
    optimal["total_system_costs"] *= 1/1000

    optimal.rename(columns={
        'ex_el_fraction': "Ext. Electr.\nFraction [1]",
        'ex_heat_fraction': "Ext. Heat.\nFraction [1]",
        'total_co2': "Total CO2\nEmissions [tonCO2]",
        'total_system_costs': "Total Costs\n[EUR]"}, inplace=True)

    grid = sns.PairGrid(data=optimal)
    grid.fig.set_size_inches(12, 8)

    grid = grid.map_upper(plt.scatter, s=2, alpha=0.1)
    # grid = grid.map_upper(sns.scatterplot, size=sizes_**2, sizes=(5,10), color = "g")
    # grid = grid.map_upper(corr)

    grid = grid.map_lower(sns.kdeplot, fill=True, cmap="BuGn")
    grid = grid.map_diag(sns.histplot, kde=True,  line_kws={
        'color': 'g'}, bins=40, edgecolor='w')

    plt.subplots_adjust(wspace=0.05)
    filename = os.path.join(output_folder, "pairplot_kpi.png")
    grid.savefig(filename, dpi=900, bbox_inches='tight')


def plot_tech_pairgrid(results, output_folder):

    optimal = results[results["Gen"] > 0]

    optimal = optimal[optimal["total_co2"] < 1e9]
    optimal = optimal[[
        'PV', 'Wind', 'P2H', 'Sto',
        ]]
    
    optimal.rename(columns={
        'PV': "PV Power\nInstalled [kW]",
        'Wind': "Wind Power\n Installed [kW]",
        'P2H': "P2H Installed [kW]",
        'Sto': "Storage Capacity\nInstalled [kWh]",
        }, 
        inplace=True)

    grid = sns.PairGrid(data=optimal)
    grid.fig.set_size_inches(12, 8)


    grid = grid.map_upper(plt.scatter, s=2, alpha=0.1)


    grid = grid.map_lower(sns.kdeplot, fill=True, cmap="BuGn")
    grid = grid.map_diag(sns.histplot, kde=True,  line_kws={
        'color': 'g'}, bins=40, edgecolor='w')

    plt.subplots_adjust(wspace=0.05)
    filename = os.path.join(output_folder, "pairplot_techs.png")
    grid.savefig(filename, dpi=900, bbox_inches='tight')


def plot_pareto(results, output_folder):
    df_last= results[results["Gen"]==max(results["Gen"])]
    pareto = get_pareto_front(df_last[["total_co2", "total_system_costs"]])

    df_pareto = deepcopy(df_last)
    df_pareto["pareto"] =False
    for idx,row in df_pareto.iterrows():
        if row["total_co2"] in [x for x in pareto["total_co2"]]:
            if row["total_system_costs"] in [y for y in pareto["total_system_costs"]]:
                df_pareto.at[idx,"pareto"]=True
    df_pareto["pareto"]

    df_pareto = df_pareto[df_pareto["pareto"]==True]
    df_pareto["total_system_costs"]*=1/1000                    

    fig,axs= plt.subplots(2,2,figsize=(10,8))
    cols = [ 'PV', 'Wind', 'P2H', 'Sto',        ]
    ax_= [axs[i][j] for i in range(2) for j in range(2)]

    for i in range(4):
        col =  cols[i]
        ax=ax_[i]
        sns.scatterplot(data=df_pareto,x="total_system_costs", y = "total_co2", hue=col, ax=ax, palette="viridis")
        ax.set_xlabel("Total Costs [EUR]")
        ax.set_ylabel("Total CO2 Emissions [Ton CO2]")

    fig.suptitle("Sizing of the Optimal System Configuration", fontweight="bold", fontsize=14, y=0.92)

    filename = os.path.join(output_folder, "sizing.png")
    fig.savefig(filename, dpi=900, bbox_inches='tight')

    fig,ax= plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df_pareto, x="total_system_costs", y= "total_co2")

    # Prepare the data for linear regression
    X = [log(x) for x in df_pareto['total_system_costs']]
    y = [y for y in df_pareto['total_co2']]
    X = sm.add_constant(X)  # Add a constant term for the intercept

    # Create and fit the linear regression model
    model = sm.OLS(y, X)
    results = model.fit()

    # Get the parameters
    intercept = results.params[0]
    slope = results.params[1]

    fit = {"x": np.linspace(0.01,max(df_pareto['total_system_costs']),200),
        "y": [intercept +slope*log(x) for x in np.linspace(0.1,max(df_pareto['total_system_costs']),200)]}
    df_fit=pd.DataFrame.from_dict(fit)
    sns.lineplot(data=df_fit, x="x", y="y", ax=ax, linestyle="--", color = "g")
    ax.set_xlim(left=0.8*min(df_pareto['total_system_costs']), right=1.01*max(df_pareto['total_system_costs']))
    ax.set_ylim(bottom=0.8*min(df_pareto['total_co2']), top= 1.01*max(df_pareto['total_co2']))

    ax.set_xlabel("Total Costs [1000 EUR]")
    ax.set_ylabel("Total CO2 Emissions [Ton CO2]")
    filename = os.path.join(output_folder, "correl.png")
    m = abs(round(slope,2))
    b= round(intercept,2)
    text = f"y = {b} -{m}*log(x)"
    ax.set_title("Correlation Between Indicators", fontweight="bold", fontsize=14)
    ax.text(0.75*max(df_pareto['total_system_costs']), 0.75*max(df_pareto['total_co2']), text, ha='center')
    fig.savefig(filename, dpi=900, bbox_inches='tight')



def evaluate_results(filename = "results_year.csv",
                     output_folder= RESULTS_YEAR_DIR):

    results = pd.read_csv(os.path.join(RESULTS_OPT_DIR,filename))
    # Plot the evolution of the points obtained
    plot_all_solutions(results, output_folder)
    plot_kpi_pairgrid(results, output_folder)
    plot_tech_pairgrid(results, output_folder)
    plot_pareto(results, output_folder)
    logging.info(f"All results plotted in: {output_folder}")
    return None

if __name__=="__main__":
    evaluate_results("results_year.csv", output_folder=RESULTS_YEAR_DIR)