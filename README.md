# PhenoScore
PhenoScore is an open source, artificial intelligence-based phenomics framework that combines facial recognition technology with Human Phenotype Ontology (HPO) data analysis to quantify phenotypic similarity at both the level of individual patients as well as of cohorts.

# Installing PhenoScore

To install PhenoScore using conda, it is advisable (but not strictly necessary) to create a new environment first:

`conda create -n PhenoScore python=3.9`

Then, activate the environment

`conda activate PhenoScore`

Install the needed dependencies using the requirements file:

`pip install -r requirements.txt`

## Optional:
To use the GPU (about a 10x performance increase when doing the LIME predictions) when CUDA is installed:

`conda install cudnn`

# Running PhenoScore

To run the basic example analysis (looking at the two subgroups in _SATB1_), just run

`python3 run_analysis.py`

after installing PhenoScore. The permutation test will be performed and corresponding LIME images generated.
Running PhenoScore on your own data is then easy: just point the PhenoScorer class to your own excel file with a similar data structure (same columns as `random_generated_sample_data.xlsx` and `satb1_data.xlsxs`.)

It is possible to run PhenoScore using three different sources of data: facial images, clinical features in HPO or both.
This can be set using the `mode` variable of the PhenoScore class to either `face`, `hpo` or `both`.

