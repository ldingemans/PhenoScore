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
        self._hpo_ids = hpo_ids
        try:
            self._simscorer = SimScorer(
                similarity_data_path=os.path.join('phenoscore', 'hpo_phenotype'),
                hpo_network_csv_path=os.path.join('phenoscore', 'hpo_phenotype', 'hpo_network.csv'),
                name_to_id_json=os.path.join('phenoscore', 'hpo_phenotype', 'hpo_name_to_id_and_reverse.json')
            )
        except:
            self._simscorer = SimScorer(
                similarity_data_path=os.path.join('..', 'hpo_phenotype'),
                hpo_network_csv_path=os.path.join('..', 'hpo_phenotype', 'hpo_network.csv'),
                name_to_id_json=os.path.join('..', 'hpo_phenotype', 'hpo_name_to_id_and_reverse.json')
            )

    def test_sim_scorer_calculations(self):
        sim_mat = np.zeros((len(self._hpo_ids), len(self._hpo_ids)))
        for i in range(len(self._hpo_ids)):
            for y in range(len(self._hpo_ids)):
                sim_mat[i, y] = self._simscorer.calc_similarity(self._hpo_ids[i], self._hpo_ids[y])
        try:
            result_csv = pd.read_csv('sim_mat_random_data.csv')
        except:
            result_csv = pd.read_csv(os.path.join('phenoscore', 'tests', 'sim_mat_random_data.csv'))
        np.testing.assert_array_almost_equal(result_csv, sim_mat, decimal=5)

    def test_sim_scorer(self):
        sim_mat = self._simscorer.calc_full_sim_mat(self._hpo_ids)
        try:
            result_csv = pd.read_csv('sim_mat_random_data_filtered_hpo.csv')
        except:
            result_csv = pd.read_csv(os.path.join('phenoscore', 'tests', 'sim_mat_random_data_filtered_hpo.csv'))
        np.testing.assert_array_almost_equal(result_csv, sim_mat, decimal=5)

    def test_filter_hpo_df(self):
        hpos = pd.DataFrame()
        hpos['hpo_all'] = ''
        hpos.at[0,'hpo_all'] = ['HP:0011927', 'HP:0000708', 'HP:0000735', 'HP:0000729', 'HP:0008771', 'HP:0001250']
        filtered_hpo = self._simscorer.filter_hpo_df(hpos)
        assert filtered_hpo.loc[0, 'hpo_all'] == [1250]

