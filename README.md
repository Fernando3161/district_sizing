# District Sizing Tool for Energy Supply

![District Sizing Tool](https://github.com/Fernando3161/district_sizing/blob/main/notebooks/sizing.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

## Introduction

District Sizing is a powerful tool developed to find the optimal sizing of devices for energy supply within an energy system. The project utilizes Multi-Objective Optimization (MOO) and oemof, a Python-based library for energy system modeling. The tool provides sizing recommendations for various technologies, including Photovoltaics (PV), Wind, Energy Storage, and Power-to-Heat (P2H) technologies, tailored specifically for residential districts.

Whether you are planning to design a new energy system for a residential area or seeking to optimize an existing one, District Sizing offers a user-friendly interface to analyze and determine the best combination of devices to ensure an efficient and sustainable energy supply.

## Features

- Perform Multi-Objective Optimization (MOO) to find the optimal device sizes for energy supply.
- Support for Photovoltaics (PV) technology to harness solar energy.
- Integration of Wind technology for utilizing wind power.
- Incorporate Energy Storage solutions to balance energy demand and supply.
- Utilize Power-to-Heat (P2H) technologies to convert excess electricity into heat.
- Interactive and intuitive interface for easy configuration and analysis.

## Installation

To install District Sizing, follow these steps:

1. Clone the repository:

```
git clone https://github.com/Fernando3161/district_sizing.git

```


2. Change directory to the project folder:


```
cd district_sizing
```

3. Install the required dependencies:

```
pip install -r requirements.txt

```


## Usage

To use the District Sizing Tool, follow these instructions:

1. Ensure you have completed the installation steps mentioned above.

2. Run the main.py script:

```
python main.py

```



3. The tool will prompt you to configure the parameters of your energy system, including the district's energy demand, available area for PV and Wind installations, and storage capacity requirements.

4. Once you've provided the necessary inputs, the optimization process will begin to find the optimal sizes for PV, Wind, Storage, and P2H technologies.

5. After the optimization process is complete, the tool will present the results, including recommended sizes for each device and additional insights.

6. Analyze the results and make informed decisions for your energy system's design and optimization.

## Notebooks

The project includes Jupyter notebooks under the `/notebooks` folder. These notebooks provide valuable insights into the development and analysis of the District Sizing Tool. The following notebooks are available:

1. [Notebook 1: Data Preprocessing](notebooks/data_preprocessing.ipynb) - Explains how the input data is preprocessed and made ready for the optimization process.

2. [Notebook 2: Optimization Process](notebooks/optimization_process.ipynb) - Details the MOO algorithm and its application in sizing energy devices for the residential district.

3. [Notebook 3: Results Analysis](notebooks/results_analysis.ipynb) - Analyses the optimization results and provides visualization for better understanding.

![District Sizing Notebook](https://github.com/Fernando3161/district_sizing/blob/main/images/district_sizing_notebook.png)

## Contributing

We welcome contributions to enhance the functionality and usability of the District Sizing Tool. If you wish to contribute, please follow the guidelines outlined in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

District Sizing is licensed under the [MIT License](LICENSE).

---

Thank you for using District Sizing! We hope this tool helps you in designing efficient and sustainable energy systems for residential districts. If you have any questions or feedback, please feel free to reach out to us. Happy optimizing!






