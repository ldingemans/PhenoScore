import os
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix is not None:
    os.environ['HOME'] = conda_prefix
else:
    os.environ['HOME'] = os.getcwd()
import unittest
from phenoscore.hpo_phenotype.calc_hpo_sim import SimScorer
import pandas as pd
import numpy as np
import ast


class SimScorerTester(unittest.TestCase):
    def setUp(self):
        try:
            random_data_csv = pd.read_excel(os.path.join('phenoscore', 'sample_data', 'random_generated_sample_data.xlsx'))
        except:
            random_data_csv = pd.read_excel(os.path.join('..', 'sample_data', 'random_generated_sample_data.xlsx'))
        hpo_ids = []
        for i in range(len(random_data_csv)):
            hpo_ids.append(ast.literal_eval(random_data_csv.loc[i, 'hpo_all']))
        self._hpo_ids = np.array(hpo_ids, dtype=object).reshape(-1, 1)
        self._simscorer = SimScorer(scoring_method='Resnik', sum_method='BMA')

    def test_get_graph(self):
        graph = self._simscorer.get_graph(['HP:0001250', 'HP:0020207'], True)
        list_of_nodes = list(graph.nodes())
        list_of_nodes.sort()
        assert list_of_nodes == ['HP:0000001', 'HP:0000118', 'HP:0000707', 'HP:0001250', 'HP:0012638', 'HP:0020207']

    def test_get_graph_labels(self):
        graph = self._simscorer.get_graph(['HP:0001250', 'HP:0020207'],  False)
        list_of_nodes = list(graph.nodes())
        list_of_nodes.sort()
        assert list_of_nodes == ['Abnormal nervous system physiology', 'Abnormality of the nervous system', 'All', 'Phenotypic abnormality', 'Reflex seizure', 'Seizure']

    def test_sim_scorer(self):
        sim_mat = self._simscorer.calc_full_sim_mat(self._hpo_ids)
        try:
            result_csv = pd.read_csv('sim_mat_random_data.csv')
        except:
            result_csv = pd.read_csv(os.path.join('phenoscore', 'tests', 'sim_mat_random_data.csv'))
        np.testing.assert_array_almost_equal(result_csv, sim_mat, decimal=0)

    def test_filter_hpo_df(self):
        hpos = pd.DataFrame()
        hpos['hpo_all'] = ''
        hpos.at[0,'hpo_all'] = ['HP:0011927', 'HP:0000708', 'HP:0000709', 'HP:0008771', 'HP:0001250']
        filtered_hpo = self._simscorer.filter_hpo_df(hpos)
        print("BLABLA" + str(filtered_hpo))
        assert filtered_hpo.loc[0, 'hpo_all'] == ['HP:0001250']

