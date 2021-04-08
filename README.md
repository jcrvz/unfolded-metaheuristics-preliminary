# Unfolded Metaheuristics - Preliminary Results

This repository contains the resulting datasets of the preliminary study on unfolded metaheuristics presented in CEC-21. We also included the main python file to plot the figures thereby presented. 

## Requirements
- Python v3.7.x
- [CUSTOMHyS framework](https://github.com/jcrvz/customhys.git)
- Standard modules: os, matplotlib, seaborn, numpy, pandas, scipy.stats

## Files
- Main script: [paper_unfolding_popvar.py](./paper_unfolding_popvar.py)
- Results for population size of 30 agents: [data_files/unfolded_hhs_pop30.json](./data_files/unfolded_hhs_pop30.json)
- Results for population size of 50 agents: [data_files/unfolded_hhs_pop50.json](./data_files/unfolded_hhs_pop50.json)
- Results for population size of 100 agents: [data_files/unfolded_hhs_pop100.json](./data_files/unfolded_hhs_pop100.json)
- Results for basic metaheuristics: [data_files/basic-metaheuristics-data_v2.json](./data_files/basic-metaheuristics-data_v2.json)
- Collection of basic metaheuristics: [collections/basicmetaheuristics.txt](./collections/basicmetaheuristics.txt)
- Collection of default heuristics: [collections/default.txt](./collections/default.txt)

## Structure of datasets

The experiments were saved in JSON files. [Further information can be found here](https://www.sciencedirect.com/science/article/pii/S2352711020303411). The data structure of saved files follows the particular scheme described below.

<details>
<summary> Expand structure </summary>
<p>

```
|-- {dict: 3}
|  |-- problem = {list: 428}
|  |  |-- 0 = {str}
:  :  :  
|  |-- dimensions = {list: 428}
|  |  |-- 0 = {int}
:  :  :  
|  |-- results = {list: 428}
|  |  |-- 0 = {dict: 5}
|  |  |  |-- step = {list: 12}
|  |  |  |  |-- 0 = {int}
:  :  :  :  :  
|  |  |  |-- performance = {list: 12}
|  |  |  |  |-- 0 = {float}
:  :  :  :  :  
|  |  |  |-- statistics = {list: 12}
|  |  |  |  |-- 0 = {dict: 10}
|  |  |  |  |  |-- nob = {int}
|  |  |  |  |  |-- Min = {float}
|  |  |  |  |  |-- Max = {float}
|  |  |  |  |  |-- Avg = {float}
|  |  |  |  |  |-- Std = {float}
|  |  |  |  |  |-- Skw = {float}
|  |  |  |  |  |-- Kur = {float}
|  |  |  |  |  |-- IQR = {float}
|  |  |  |  |  |-- Med = {float}
|  |  |  |  |  |-- MAD = {float}
:  :  :  :  :  
|  |  |  |-- encoded_solution = {list: 12}
|  |  |  |  |-- 0 = {list: 50}
|  |  |  |  |  |-- 0 = {int}
:  :  :  :  :  :  
:  :  :  :  :  
|  |  |  |-- hist_fitness = {list: 50}
|  |  |  |  |-- 0 = {list: 98}
|  |  |  |  |  |-- 0 = {float}
:  :  :  :  :  :  
:  :  :  :  :  
:  :  :  
```
</p>
</details>
