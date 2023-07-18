import unittest
import sys
import os
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)

from src.prepare_district_data import prepare_district_dataframe
from src.district import Scenarios, create_energy_system


BOUNDARY_DATA = prepare_district_dataframe(year=2020,days=7, start_day=0)


class TestCreateEnergySystem(unittest.TestCase):

    def test_create_energy_system_default_sizing(self):
        # Test if the energy system is created correctly with default sizing
        boundary_data = BOUNDARY_DATA
        for scenario in Scenarios:
            energy_system = create_energy_system(boundary_data, scenario)
            assert(energy_system is not None)

    def test_create_energy_system_custom_sizing(self):
        # Test if the energy system is created correctly with default sizing
        boundary_data = BOUNDARY_DATA
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

        for scenario in Scenarios:
            energy_system = create_energy_system(boundary_data, scenario, sizing=sizing)
            assert(energy_system is not None)

if __name__ == "__main__":
    unittest.main()
