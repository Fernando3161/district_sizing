'''
Created on 29.06.2021

@author: Fernando Penaherrera @UOL/OFFIS

Creates a model with demands for a district and several power plants.

There are 4 different scenarions with prices inflated to show preference

for different markets (The baseline scenario shows strong preference towards
the Intradaymarket

The Energy System of the District consists of:

* PV System
* Electric Storage
* Gas Boiler
* CHP
* Thermal Storage
* Electric Grid
* Gas Grid
* Electric bus for selling energy
'''

from pprint import pprint
from constants import EMISSION_GAS, EMMISION_ELE, EUR
from financial import calculate_annualized_npv
from prepare_district_data import prepare_district_dataframe
from common import RESULTS_DIR
import warnings
import pandas as pd
import numpy as np
from oemof.solph import EnergySystem, Bus, Flow
from oemof.solph.components import Sink, Source, Transformer, GenericStorage
from oemof.solph import views, processing
import matplotlib.pyplot as plt
import logging
from enum import Enum
from os.path import join
from oemof.solph import Model
import seaborn as sns
sns.set_theme(style="darkgrid", palette="deep", font="arial")


warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)


class Scenarios(Enum):
    '''
    Each scenario represents wich price market has been artificially inflated
    '''
    NO_RE = 0
    PV = 1
    WIND = 2
    PV_WIND = 3


def create_energy_system(boundary_data, scenario, sizing=None):
    # Default Data of the devices of the disctrit
    # The same configuration needs to be passed if changes are to be made
    # to the district configuraiton
    if sizing is None:
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
            "Capacity": 200,  # kWh
            "Eff_Inflow": 0.98,  # 1
            "Eff_Outflow": 0.98,  # 1
        }
        sizing["CHP"] = {
            "ElectricPower": 30,  # kW
            "ThermalPower": 60,  # kW
            "ElectricEfficiency": 0.3,  # 1
            "ThermalEfficiency": 0.6,  # 1
        }
    boundary_data = boundary_data.asfreq('H')
    # Create Energy System with the dataframe time series
    energy_system = EnergySystem(timeindex=boundary_data.index)

    # Buses
    b_renewable = Bus(label="b_renewable")
    b_el_out = Bus(label="b_el_out", inputs={b_renewable: Flow()})
    b_electric_supply = Bus(label="b_electric_supply",
                            inputs={b_renewable: Flow()})
    b_heat_gas = Bus(label="b_gas")
    b_heat_supply = Bus(label="b_heat_supply")

    energy_system.add(
        b_electric_supply,
        b_renewable,
        b_el_out,
        b_heat_gas,
        b_heat_supply)

    # Energy Sources
    s_electric_grid = Source(
        label="s_electric_grid",
        outputs={
            b_electric_supply: Flow(
                variable_costs=boundary_data["Elec_price"] / 1000)})  # EUR/kWh

    s_gas = Source(
        label='m_gas',
        outputs={
            b_heat_gas: Flow(
                variable_costs=boundary_data["Gas_price"] / 1000)})  # EUR/kWh

    # Create local energy demand
    d_el = Sink(label='d_el',
                inputs={
                    b_electric_supply: Flow(
                        fix=boundary_data['Electricity'],
                        nominal_value=1
                    )})

    d_heat = Sink(label='d_heat',
                  inputs={
                      b_heat_supply: Flow(
                          fix=boundary_data['Heat'],
                          nominal_value=1
                      )})

    energy_system.add(s_electric_grid, s_gas, d_el, d_heat)

    # Technologies
    # Photovoltaic
    if scenario in [Scenarios.PV, Scenarios.PV_WIND]:
        s_pv = Source(
            label="s_pv",
            outputs={
                b_renewable: Flow(
                    nominal_value=1,
                    fix=boundary_data["PV_pu"]*sizing["PV"])})
        energy_system.add(s_pv)

    if scenario in [Scenarios.WIND, Scenarios.PV_WIND]:

        s_wind = Source(
            label="s_wind",
            outputs={
                b_renewable: Flow(
                    nominal_value=1,
                    fix=boundary_data["Wind_pu"]*sizing["WT"]
                )})
        energy_system.add(s_wind)

    # Boiler
    t_boiler = Transformer(
        label='t_boiler',
        inputs={b_heat_gas: Flow()},
        outputs={
            b_heat_supply: Flow(nominal_value=sizing["Boiler"]['Power'])},
        conversion_factors={
            b_heat_gas: 1,
            b_heat_supply: sizing["Boiler"]['Eff']})

    # Electric Battery
    sto_battery = GenericStorage(
        label='sto_battery',
        inputs={
            b_renewable: Flow(nominal_value=sizing["Battery"]["Input_Power"])},
        outputs={
            b_renewable: Flow(nominal_value=sizing["Battery"]["Output_Power"])},
        loss_rate=sizing["Battery"]["Self_Discharge"],
        nominal_storage_capacity=sizing["Battery"]["Capacity"],
        inflow_conversion_factor=sizing["Battery"]["Eff_Inflow"],
        outflow_conversion_factor=sizing["Battery"]["Eff_Outflow"],
        initial_storage_level=0,
        balanced=False)

    # CHP
    t_p2h = Transformer(
        label="t_p2h",
        inputs={b_electric_supply: Flow(
            variable_costs=boundary_data["Elec_price"] / 1000)},

        outputs={b_heat_supply: Flow(nominal_value=sizing["P2H"])},
        conversion_factors={
            b_electric_supply: 1,
            b_heat_supply: 1})

    # t_chp = Transformer(
    #     label='t_chp',
    #     inputs={b_heat_gas: Flow()},
    #     outputs={
    #         b_electric_supply: Flow(nominal_value=sizing["CHP"]["ThermalPower"]),
    #         b_heat_supply: Flow(nominal_value=sizing["CHP"]["ElectricPower"])},
    #     conversion_factors={
    #         b_renewable: sizing["CHP"]["ElectricEfficiency"],
    #         b_heat_supply: sizing["CHP"]["ThermalEfficiency"]})
    # energy_system.add(t_chp)

    if sizing["Boiler"]['Power'] + sizing["P2H"] < boundary_data["Heat"].max():
        raise AssertionError(
            "Thermal power not enough for district. Cheack Boiler and CHP Sizes")

    energy_system.add(t_boiler, sto_battery, t_p2h)

    # Markets. Prices are in EUR/kWh for consistency.
    s_day_ahead = Sink(
        label="s_da",
        inputs={
            b_el_out: Flow(
                variable_costs=-
                boundary_data["Elec_price"]/1001)})

    energy_system.add(s_day_ahead)

    return energy_system


def solve_model(model):
    '''
    Solve the constrained model

    :param model: oemof.solph model.
    '''
    # Solve the model
    model.solve(solver="glpk", solve_kwargs={"tee": False})

    energy_system = model.es
    if model.solver_results.Solver[0].Status != "ok":
        raise AssertionError("Solver did not converge. Stopping simulation")

    energy_system.results['valid'] = True
    energy_system.results['solve_and_write_data'] = processing.results(
        model)
    energy_system.results['solve_and_write_data'] = views.convert_keys_to_strings(
        energy_system.results['solve_and_write_data'])
    energy_system.results['meta'] = processing.meta_results(
        model)

    return energy_system


def post_process_results(energy_system):
    '''
    Process Results into a nicer Data Frame

    :param energy_system: Solved energy system
    '''

    results = energy_system.results['solve_and_write_data']
    results_list = []
    for k in results.keys():
        if "flow" in list(results[k]['sequences'].keys()):
            flow = results[k]['sequences']['flow']
            if True:
                key_name = str(k)
                for s in [
                        "(", "'", ")"]:  # remove ( ' ) characters
                    key_name = key_name.replace(s, "")
                flow.rename(key_name, inplace=True)
                flow_df = pd.DataFrame(flow)
                results_list.append(flow_df)
    results = pd.concat(results_list, axis=1)

    return results


def save_plot_results(results, year, scenario):
    '''
    Save the results in cleaner dataframes and in a graphic

    :param results: Results dataframe
    :param year: Year of the analysis
    :param scenario: Scenario number
    '''
    columns = results.columns

    # Get the results of selling energy
    res_sell = [c for c in columns if "b_el_out" in c.split(",")[0]]
    data_plot = results[res_sell]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data_plot, ax=ax)
    ax.set_title("Energy Sold")
    # Create a plot

    results.to_csv(join(RESULTS_DIR,
                        "MarketResults{}-Sc{}.csv".format(year, scenario.value)))
    fig.savefig(join(RESULTS_DIR,
                     "MarketResults{}-Sc{}.jpg".format(year, scenario.value)))
    logging.info(
        f"Results saved for year {year} and Scenario {scenario.value}")


def calculate_kpis(results, boundary_data, sizing, scenario):
    # Calculate CO2
    kpis = {}
    total_d_gas = results["m_gas, b_gas"]
    total_d_ele = results["s_electric_grid, b_electric_supply"]
    total_sell_ele = results["b_el_out, s_da"]

    total_use_re = results["b_renewable, b_electric_supply"].sum()
    total_demand_ele = results["b_electric_supply, d_el"].sum()
    total_p2h = results["b_electric_supply, t_p2h"].sum()
    re_fraction = total_use_re/(total_demand_ele+total_p2h)
    kpis["re_fraction"] = re_fraction

    total_external_electricity = results["s_electric_grid, b_electric_supply"].sum(
    )
    external_el_fraction = total_external_electricity / \
        (total_demand_ele+total_p2h)
    kpis["ex_el_fraction"] = external_el_fraction

    total_boiler_heat = results["t_boiler, b_heat_supply"].sum()
    total_heat_demand = results["b_heat_supply, d_heat"].sum()
    external_heat_fraction = total_boiler_heat/total_heat_demand.sum()
    kpis["ex_heat_fraction"] = external_heat_fraction

    kpis["p2h_heat_fraction"] = 1-external_heat_fraction

    total_co2 = total_d_gas.sum()*EMISSION_GAS + total_d_ele.sum()*EMMISION_ELE
    total_co2 *= 1e-6  # ton CO2

    cost_ele = total_d_ele*boundary_data["Elec_price"]
    sell_ele = total_sell_ele*boundary_data["Elec_price"]
    cost_gas = total_d_gas*boundary_data["Gas_price"]
    total_cost = cost_ele.sum() - sell_ele.sum()+cost_gas.sum()
    total_cost *= 1/1000  # EUR
    # investments:
    pv_size = 0
    if scenario in [Scenarios.PV, Scenarios.PV_WIND]:
        pv_size = sizing["PV"]   # kW
    wt_size = 0
    if scenario in [Scenarios.WIND, Scenarios.PV_WIND]:
        wt_size = sizing["WT"]  # kW
    boiler_size = sizing["Boiler"]["Power"]  # kW
    p2h_size = sizing["P2H"]  # kW
    batt_size = sizing["Battery"]["Capacity"]  # kWh

    annual_table = pd.DataFrame()

    annual_table = pd.DataFrame()
    annual_table["Tech"] = ["pv", "wt", "boiler", "p2h", "batt"]
    annual_table.set_index("Tech", inplace=True)
    annual_table["size"] = [pv_size, wt_size, boiler_size, p2h_size, batt_size]
    annual_table["cost_per_kw"] = [875/EUR, 1500, 8625/113, 150, 151]
    annual_table["total_cost"] = annual_table["size"] * \
        annual_table["cost_per_kw"]
    annual_table["lifetime"] = [20, 20, 10, 10, 10]
    annual_table["repair_cost"] = [0.3*x for x in annual_table["total_cost"]]
    annual_table["repair_time"] = [5, 5, 3, 3, 5]
    annual_table["energy_deprec"] = [0.2/10, 0.2/10, 0.1/10, 0.1/10, 0.3/10]
    annual_table["financial_deprec"] = [5/100]*5
    annual_table["NPV"] = np.nan
    no_of_days = boundary_data.shape[0]/24
    annual_table["NPV_days"] = np.nan

    for idx, row in annual_table.iterrows():
        if row["total_cost"] == 0:
            annual_table.at[idx, "NPV"] = 0
            annual_table.at[idx, "NPV_days"] = 0
        else:
            anpv = calculate_annualized_npv(capital_cost=row["total_cost"],
                                            lifetime=int(row["lifetime"]),
                                            repair_cost=row["repair_cost"],
                                            repair_interval=int(
                                                row["repair_time"]),
                                            discount_rate=row["energy_deprec"]+row["financial_deprec"])

            annual_table.at[idx, "NPV"] = -anpv
            annual_table.at[idx, "NPV_days"] = -anpv*no_of_days/365

    total_invest_cost = annual_table["NPV_days"].sum()

    total_system_costs = total_cost + total_invest_cost

    kpis["total_co2"] = total_co2
    kpis["energy_cost"] = total_cost
    kpis["total_system_costs"] = total_system_costs

    return kpis


def create_and_solve_scenario(start_day=None, days=1, year=2017, sizing=None, scenario=Scenarios.NO_RE):
    '''
    Chain of functions to model the different scenarios

    :param days: Number of days
    :param year: Year of simulation
    :param sizing: Sizing data. Can be empty and default data will be passed
    :param scenario: Scenario Enum value
    '''
    boundary_data_year = prepare_district_dataframe(year=year)
    if start_day:

        boundary_data = boundary_data_year.tail(365*34-start_day*24)
    else:
        boundary_data = boundary_data_year
    boundary_data = boundary_data.head(days * 24)
    energy_system = create_energy_system(
        boundary_data, scenario=scenario, sizing=sizing)
    model = Model(energy_system)
    solved_energy_system = solve_model(model)
    results = post_process_results(solved_energy_system)
    save_plot_results(results, year, scenario)
    kpis = calculate_kpis(results, boundary_data, sizing, scenario=scenario)
    return kpis


def main(sizing=None, year=2019, days=28, start_day=None):
    for scenario in Scenarios:
        kpis = create_and_solve_scenario(
            start_day=start_day, days=days, year=year, sizing=sizing, scenario=scenario)
        pprint(kpis)
    logging.info("All scenarios terminated succesfully")


if __name__ == '__main__':
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

    main(sizing=sizing, year=2020, days=7, start_day=90)
