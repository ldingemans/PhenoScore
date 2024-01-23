# PhenoScore
![PhenoScore logo](phenoscore_logo.png) 
PhenoScore is an open source, artificial intelligence-based phenomics framework that combines facial recognition technology with Human Phenotype Ontology (HPO) data analysis to quantify phenotypic similarity at both the level of individual patients as well as of cohorts.

# Installing PhenoScore

To install PhenoScore using conda, it is advisable (but not strictly necessary) to create a new environment first:

`conda create -n PhenoScore python=3.9`

Then, activate the environment

`conda activate PhenoScore`

Install PhenoScore and the needed dependencies:

`pip install .`

To fix some dependency issues, especially on Windows, please then run these commands:

`pip install mxnet==1.7.0post2 --no-dependencies`

`pip install numpy==1.23.5`

# Optional:
To use the GPU (about a 10x performance increase when doing the LIME predictions) when CUDA is installed:

`conda install cudnn`

# Running PhenoScore

To run the basic example analysis (looking at the two subgroups in _SATB1_), take a look at the tutorial in the ``notebooks`` directory. 

