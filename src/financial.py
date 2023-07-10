"""
Calculates the annual net present value

Done with ChatGPT
"""


def calculate_annualized_npv(capital_cost, lifetime, repair_cost, repair_interval, discount_rate):
    """
    write a python code to calculate the annualized net present value value of an investment
    of a machine with an initial capital cost of A, lifetime of t. 
    The machine must be reparated with a repair cost of C  every n years.

    """
    cash_flows = [-capital_cost]  # Initial investment (negative cash flow)

    # Generate cash flows for each repair interval
    for year in range(repair_interval, lifetime + 1, repair_interval):
        cash_flows.append(-repair_cost)  # Repair cost (negative cash flow)

    npv = 0

    for year, cash_flow in enumerate(cash_flows):
        discounted_cash_flow = cash_flow / ((1 + discount_rate) ** year)
        npv += discounted_cash_flow

    # Calculate the annualized net present value
    annualized_npv = npv * (discount_rate / (1 - (1 + discount_rate) ** -lifetime))

    return annualized_npv