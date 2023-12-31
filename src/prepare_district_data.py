
import sys
import os
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)

import pandas as pd
from os.path import join
from src.common import DATA_DIR

GAS_PRICE = 3.66  # EUR per mmBTU
# Convert GAS_PRICE to EUR/MWh
# EUR/MWh = EUR/mmBTU / (293.07 kWh/mmBTU)*(1000 kWh/MWh)
GAS_PRICE = GAS_PRICE / 293.07 * 1000

def prepare_district_dataframe(year=2017, days = 30, start_day=0):
    '''
    Builds a dataframe with the information of the district found
    in the first excel file found. By default, searches for the year 2017.

    Args:
        year (int): Year for which the data is prepared.

    Returns:
        pandas.DataFrame: Dataframe with the district energy demand
                          information.
    '''

    EXCEL_DATA = join(DATA_DIR, "quartier1_2017.xlsx")

    start = str(year) + "-01-01"
    days_year = 365  # Full year

    # Correct for leap year
    if year % 4 == 0:
        days_year += 1

    # Set datetime objects with appropriate dates
    dates = pd.date_range(start, periods=days_year * 24 + 1, freq="H")

    # Electricity
    electricity = pd.read_excel(EXCEL_DATA,
                                "electricity demand series",
                                engine='openpyxl')["DE01"][2:].tolist()

    # Heat
    heat = pd.read_excel(EXCEL_DATA, "heat demand series", engine='openpyxl')[
        "DE01"][2:].tolist()

    # PV Production per kW installed (as data source, no direct PV Modeling)
    pv_df = pd.read_excel(EXCEL_DATA, "volatile series", engine='openpyxl')
    pv_pu = pv_df[pv_df.columns[3]][2:].tolist()

    # Wind Production per kW installed (as data source, no direct WK Modeling)
    wind_df = pd.read_excel(EXCEL_DATA, "volatile series", engine='openpyxl')
    wind_pu = wind_df[wind_df.columns[4]][2:].tolist()

    # Complete the last hour to get a full year.
    # Copy the last value for simplicity
    electricity.append(electricity[-1])
    heat.append(heat[-1])
    pv_pu.append(pv_pu[-1])
    wind_pu.append(wind_pu[-1])

    # Incorporate electricity data
    electricity_data = join(DATA_DIR, "Germany.csv")
    df_electricity = pd.read_csv(electricity_data, parse_dates=True)

    if year not in range(2015, 2023):
        year = 2022

    # Filter electricity data for the specified year
    df_electricity = df_electricity[
        (df_electricity['Datetime (Local)'] > f"{year}-01-01") &
        (df_electricity['Datetime (Local)'] < f"{year+1}-01-01")]
    df_electricity = df_electricity[['Datetime (Local)', "Price (EUR/MWhe)"]]
    df_electricity.rename(columns={'Datetime (Local)': "Time",
                                   "Price (EUR/MWhe)": "Elec_Price"},
                          inplace=True)
    df_electricity.set_index("Time", inplace=True)

    # Build Dataframe from a dictionary
    min_len = min(len(dates), len(electricity), len(heat), len(pv_pu),
                  len(wind_pu), len(df_electricity["Elec_Price"]))
    district_data = {
        "Date": dates[0:min_len],
        "Electricity": electricity[0:min_len],
        "Heat": heat[0:min_len],
        "PV_pu": pv_pu[0:min_len],
        "Wind_pu": wind_pu[0:min_len],
        "Elec_price": df_electricity["Elec_Price"][0:min_len],
        "Gas_price": [GAS_PRICE for _ in range(min_len)]
    }

    district_df = pd.DataFrame.from_dict(district_data)
    district_df.set_index("Date", inplace=True)

    district_df.to_csv(join(DATA_DIR, f"district_data_{year}"))

    # Remove the last value as it is for 01-Jan 00:00 of the next year.
    

    district_df = district_df[:-1]

    if start_day:
        district_df = district_df.tail(365*34-start_day*24)
    else:
        district_df = district_df
    district_df = district_df.head(days * 24)

    return district_df

if __name__ =="__main__":
    prepare_district_dataframe(year=2017, days=8, start_day=0)
