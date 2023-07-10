import numpy as np
from pymoo.core.problem import Problem

from district import Scenarios, calculate_kpis, create_energy_system, post_process_results, solve_model
from prepare_district_data import prepare_district_dataframe
from oemof.solph import Model


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

class ProblemES(Problem):
    year = 2020
    start_day = 90
    days = 7
    boundary_data = None
    es_size = sizing
    results_ = []
    scenario = Scenarios.PV_WIND
    generation = 0


    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=2,
                         xl=0.0,
                         xu=300.0)
        
    def build_boundary_data(self):
        boundary_data_year = prepare_district_dataframe(year=self.year)
        if self.start_day:
            boundary_data = boundary_data_year.tail(365*34-self.start_day*24)
        else:
            boundary_data = boundary_data_year
        boundary_data = boundary_data.head(self.days * 24)

        self.boundary_data=boundary_data

    def _evaluate(self, x, out, *args, **kwargs):
        if self.boundary_data is None:
            self.build_boundary_data()

        f1_ = []
        f2_ = []
        for i in range(x.shape[0]):
            self.es_size["PV"] = x[i][0]  # kW
            self.es_size["WT"] = x[i][1]  # kW
            self.es_size["P2H"] = x[i][2]  # kW
            self.es_size["Battery"]["Input_Power"] = x[i][3]/10
            self.es_size["Battery"]["Output_Power"] = x[i][3]/10
            self.es_size["Battery"]["Capacity"] = x[i][3]

            energy_system = create_energy_system(
                self.boundary_data, scenario=self.scenario, sizing=self.es_size)
            model = Model(energy_system)
            solved_energy_system = solve_model(model)
            results_sim = post_process_results(solved_energy_system)
            kpis = calculate_kpis(
                results_sim, self.boundary_data, self.es_size, scenario=self.scenario)

            f1 = kpis["total_co2"]
            f2 = kpis["total_system_costs"]
            self.results_.append([self.generation, [g for g in x[i]], kpis])
            f1_.append(f1)
            f2_.append(f2)

        self.generation += 1

        out["F"] = np.column_stack([f1_, f2_])
