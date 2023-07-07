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
* 4 Sinks for selling energy representing the 4 electric markets
'''

from prepare_district_data import prepare_district_dataframe
from common import RESULTS_DIR
import warnings
import pandas as pd
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
        sizing["WT"] = 1000  # kW
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
                    nominal_value= 1,
                    fix=boundary_data["PV_pu"]*sizing["PV"] )})
        energy_system.add(s_pv)

    if scenario in [Scenarios.WIND, Scenarios.PV_WIND]:

        s_wind = Source(
            label="s_wind",
            outputs={
                b_renewable: Flow(
                    nominal_value=sizing["WT"],
                    fix=boundary_data["Wind_pu"] 
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


    if sizing["Boiler"]['Power'] + \
            sizing["CHP"]["ThermalPower"] < boundary_data["Heat"].max():
        raise AssertionError(
            "Thermal power not enough for district. Cheack Boiler and CHP Sizes")

    energy_system.add(t_boiler, sto_battery, t_p2h)

    # Markets. Prices are in EUR/kWh for consistency.
    s_day_ahead = Sink(
        label="s_da",
        inputs={
            b_el_out: Flow(
                variable_costs=-
                boundary_data["Elec_price"]/1000)})

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


def create_and_solve_scenario(days=7, year=2017, sizing=None, scenario=1):
    '''
    Chain of functions to model the different scenarios

    :param days: Number of days
    :param year: Year of simulation
    :param sizing: Sizing data. Can be empty and default data will be passed
    :param scenario: Scenario Enum value
    '''

    boundary_data = prepare_district_dataframe(year=year).head(days * 24)
    energy_system = create_energy_system(boundary_data, sizing)
    model = Model(energy_system)
    solved_energy_system = solve_model(model)
    results = post_process_results(solved_energy_system)
    save_plot_results(results, year, scenario)


def main(year=2019, days=28):
    for scenario in Scenarios:
        create_and_solve_scenario(days=days, year=year, scenario=scenario)
    logging.info("All scenarios terminated succesfully")


if __name__ == '__main__':
    main(year=2020, days=7)
