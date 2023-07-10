# %%

"""
Script for evaluating n*m configurations for sizing of PV and WT
"""
import pandas as pd
from oemof.solph import Model
from district import Scenarios, calculate_kpis, create_energy_system, post_process_results, solve_model
from prepare_district_data import prepare_district_dataframe


sizing = dict()
sizing["PV"] = 500  # kW
sizing["WT"] = 300  # kW
sizing["Boiler"] = {"Power": 300,  # kW
                    "Eff": 0.85,  # 1
                    }
sizing["P2H"] = 60  # kW
sizing["Battery"] = {
    "Input_Power": 10,  # kW
    "Output_Power": 10,  # kW
    "Self_Discharge": 0.01,  # 1
    "Capacity": 100,  # kWh
    "Eff_Inflow": 0.98,  # 1
    "Eff_Outflow": 0.98,  # 1
}
sizing["CHP"] = {
    "ElectricPower": 30,  # kW
    "ThermalPower": 60,  # kW
    "ElectricEfficiency": 0.3,  # 1
    "ThermalEfficiency": 0.6,  # 1
}

# In this example I will map as a grid the two indicators (cost, co2) based only on the sizing of the RE faclilites
year = 2020
days = 30
scenario = Scenarios.PV_WIND
boundary_data = prepare_district_dataframe(year=year).head(days * 24)
list_pv = range(0, 200, 5)
list_wind = range(0, 200, 5)

results = {}
results["PV"] = []
results["WIND"] = []
results["TotalCost"] = []
results["TotalCO2"] = []
for x in list_pv:
    for y in list_wind:
        sizing["PV"] = x  # kW
        sizing["WT"] = y  # kW

        energy_system = create_energy_system(
            boundary_data, scenario=scenario, sizing=sizing)
        model = Model(energy_system)
        solved_energy_system = solve_model(model)
        results_sim = post_process_results(solved_energy_system)
        co2, cost = calculate_kpis(
            results_sim, boundary_data, sizing, scenario=scenario)

        results["PV"].append(x)
        results["WIND"].append(y)
        results["TotalCost"].append(cost)
        results["TotalCO2"].append(co2)

df = pd.DataFrame.from_dict(results)

import os

df.to_csv(os.path.join(os.getcwd(), "iteration_results_1.csv"))
