# PhenoScore
PhenoScore is an open source, artificial intelligence-based phenomics framework that combines facial recognition technology with Human Phenotype Ontology (HPO) data analysis to quantify phenotypic similarity at both the level of individual patients as well as of cohorts.

# Installing PhenoScore

To install PhenoScore using conda, it is advisable (but not strictly necessary) to create a new environment first:

`conda create -n PhenoScore python=3.9`

Then, activate the environment

`conda activate PhenoScore`

Install the needed dependencies:

`pip install numpy pandas tqdm tensorflow==2.5 deepface sklearn obonet phenopy openpyxl seaborn`

For the LIME library, we need some special (minor) modifications, so install it from my repository:

`git clone https://github.com/ldingemans/lime.git
pip install lime/.`

Finally, download `phenotype.hpoa` from the HPO website or github (for instance from `https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2022-06-11/phenotype.hpoa`) and place it in the root.

## Optional:
To use the GPU (about a 10x performance increase when doing the LIME predictions) when CUDA is installed:

`conda install cudnn`

# Running PhenoScore

To run the basic example analysis (looking at the two subgroups in _SATB1_ and some randomly generated data, just run

`python3 run_analysis.py`

after installing PhenoScore. The permutation test will be performed and corresponding LIME images generated.
Running PhenoScore on your own data is then easy: just replace the _SATB1_ data in the sample_data directory (or change the loading of `df_data` in `run_analysis`).

It is possible to run PhenoScore using three different sources of data: facial images, clinical features in HPO or both.
This can be set using the `PHENOSCORE_MODE` variable to either `face`, `hpo` or `both`.

