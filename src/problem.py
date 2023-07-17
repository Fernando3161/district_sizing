import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from multiprocessing.pool import ThreadPool
from time import time
from oemof.solph import Model
import logging

from district import Scenarios, calculate_kpis, create_energy_system, post_process_results, solve_model
from prepare_district_data import prepare_district_dataframe

logging.basicConfig(level=logging.ERROR)


class ProblemES(Problem):
    """
    Represents the optimization problem for energy system sizing.
    """

    year: int = 2020
    start_day: int = 0
    days: int = 365
    boundary_data: pd.DataFrame = None
    es_size: np.ndarray = None
    results_: list = []
    scenario: Scenarios = Scenarios.PV_WIND
    generation: int = 0
    kpis_names: list = None
    n_indivs: int = 50

    def __init__(self) -> None:
        """
        Initializes the energy system optimization problem.
        """
        super().__init__(n_var=4, n_obj=2, xl=0.0, xu=400.0)

    def build_boundary_data(self) -> None:
        """
        Builds the boundary data for the energy system optimization.
        """
        boundary_data_year = prepare_district_dataframe(year=self.year)
        boundary_data = boundary_data_year.tail(365 * 24 - self.start_day * 24)
        boundary_data = boundary_data.head(self.days * 24)
        self.boundary_data = boundary_data

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """
        Evaluates the objective function(s) for the given solution(s).

        Parameters:
            X (np.ndarray): The decision variable values for the solutions.
            out (dict): Dictionary to store the evaluated objective values.

        Returns:
            None
        """
        start_time = time()
        if self.boundary_data is None:
            self.build_boundary_data()

        f1_ = []
        f2_ = []

        def eval_x_es(x: list) -> tuple:
            """
            Evaluates the objective function values for a single solution.

            Parameters:
                x (list): List of decision variable values for a single solution.

            Returns:
                tuple: Objective function values for the solution.
            """
            sizing = dict()
            sizing["PV"] = x[0]  # kW
            sizing["WT"] = x[1]  # kW
            sizing["Boiler"] = {
                "Power": max(300 - x[2], 10),  # kW
                "Eff": 0.85  # 1
            }
            sizing["P2H"] = x[2]  # kW
            sizing["Battery"] = {
                "Input_Power": x[3] / 10,  # kW
                "Output_Power": x[3] / 10,  # kW
                "Self_Discharge": 0.01,  # 1
                "Capacity": x[3],  # kWh
                "Eff_Inflow": 0.98,  # 1
                "Eff_Outflow": 0.98  # 1
            }

            try:
                energy_system = create_energy_system(
                    self.boundary_data, scenario=self.scenario, sizing=sizing)
                model = Model(energy_system)
                solved_energy_system = solve_model(model)
                results_sim = post_process_results(solved_energy_system)
                kpis = calculate_kpis(
                    results_sim,
                    self.boundary_data,
                    sizing=sizing,
                    scenario=self.scenario)

                f1 = kpis["total_co2"]
                f2 = kpis["total_system_costs"]
                f3 = kpis["ex_el_fraction"]

                if self.generation == 0 and self.kpis_names is None:
                    self.kpis_names = [x for x in kpis.keys()]
                kpi_val = [k for k in kpis.values()]

            except Exception as e:
                kpi_val = [1000000000] * 8  # len(self.kpis_names)
                f1 = 1000000000
                f2 = 1000000000
                f3 = 1000000000

            self.results_.append(
                [self.generation + 1] + [g for g in x] + kpi_val)
            return f1, f2, f3

        params = [[X[k]] for k in range(len(X))]

        pool = ThreadPool(self.n_indivs)
        F = pool.starmap(eval_x_es, params)

        self.generation += 1
        f1_ = [F[i][0] for i in range(len(F))]
        f2_ = [F[i][1] for i in range(len(F))]
        f3_ = [F[i][2] for i in range(len(F))]

        if self.n_obj == 2:
            out["F"] = np.column_stack([f1_, f2_])
        elif self.n_obj == 3:
            out["F"] = np.column_stack([f1_, f2_, f3_])

        pool.close()
        end_time = time()
        duration = round(end_time - start_time, 1)
        print(f"Generation {self.generation} completed in {duration} s.", end="\r")

    def finalize(self, filename: str = "data.csv") -> None:
        """
        Finalizes the optimization process and saves the results to a file.

        Parameters:
            filename (str): The name of the file to save the results.

        Returns:
            None
        """
        data = {}
        cols = ["Gen", "PV", "Wind", "P2H", "Sto"] + [x for x in self.kpis_names]
        for i in range(len(cols)):
            data[cols[i]] = [x[i] for x in self.results_]
        data_df = pd.DataFrame.from_dict(data)
        data_df.to_csv(filename)
