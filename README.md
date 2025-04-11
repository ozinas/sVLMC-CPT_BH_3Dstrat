# sVLMC-CPT_BH_3Dstrat
This set of python notebooks and functions contain the code to reproduce some results of the paper: Zinas, O., Papaioannou, I., Schneider, R., Cuéllar, P., 2025. Multivariate gaussian process regression for 3d site characterization from cpt and categorical borehole data. Engineering Geology , 108052URL: https://www.sciencedirect.com/science/article/pii/S0013795225001486, doi:https://doi.org/10.1016/j.enggeo.2025.108052.

This repository contains Jupyter notebooks utilizing **GPyTorch** for Gaussian Process modeling. The code is intended to **reproduce specific results** from our study and is not provided as a general-purpose model. It includes working scripts that were used to generate the reported results. 

## Requirements
- Python 3.12.2
- scipy 1.11.3
- GPyTorch 1.11
- BoTorch  0.9.5
- PyTorch 2.2.0+cu118
- pandas 2.1.4
- openpyxl
- matplotlib
- scikit-learn 1.4.1.post1
- Jupyter Notebook 7.0.8
- KeOps
- cloudpickle 3.0.0

  CUDA Version: 12.0

## Usage
These notebooks are shared for transparency and reproducibility. Users can run the scripts to obtain similar results, but modifications may be required for different levels of accuracy.

If you find these notebooks useful for your research, please cite the article:

@article{ZINAS2025108052,
title = {Multivariate Gaussian Process Regression for 3D site characterization from CPT and categorical borehole data},
journal = {Engineering Geology},
volume = {352},
pages = {108052},
year = {2025},
issn = {0013-7952},
doi = {https://doi.org/10.1016/j.enggeo.2025.108052},
url = {https://www.sciencedirect.com/science/article/pii/S0013795225001486},
author = {Orestis Zinas and Iason Papaioannou and Ronald Schneider and Pablo Cuéllar},
keywords = {Site characterization, Stratigraphy prediction, Multivariate Gaussian process, Linear Model of Coregionalization, Variational inference}}

## Disclaimer
This code is provided **as-is** for transparency purposes. It is not a fully developed software package and may require adjustments to work with other data or applications.

