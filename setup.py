from setuptools import setup, find_packages

# read requirements/dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# read description/README.md
with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='phenoscore',
      version='0.1.0',
      packages=find_packages(),
      install_requires=requirements,

      package_data={'': ['test_data/*']},
      long_description=long_description,
      long_description_content_type='text/markdown',

      author='ALexander JM Dingemans',
      author_email='a.dingemans@radboudumc.nl',
      url='https://github.com/ldingemans/PhenoScore',
      description='PhenoScore is an open source, artificial intelligence-based phenomics framework that combines '
                  'facial recognition technology with Human Phenotype Ontology (HPO) data analysis to quantify '
                  'phenotypic similarity at both the level of individual patients as well as of cohorts.',
      license='GNU General Public License v3.0',
      keywords='machine learning, artificial intelligence, genetics, clinical genetics',

      # entry_points={'console_scripts': [
          # 'c2s2 = c2s2.main:main'
      # ]}
      )