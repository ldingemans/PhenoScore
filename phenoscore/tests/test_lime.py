import unittest
from phenoscore.phenoscorer import PhenoScorer
import os


# WARNING: this test does a full run and takes a while to run on a CPU


class CrossValLIMETester(unittest.TestCase):
    def setUp(self):
        self._phenoscorer = PhenoScorer(gene_name='SATB1', mode='both', method_hpo_similarity='Resnik',
                                        method_summ_hpo_similarities='BMA')

    def test_lime_gen(self):
        X, y, img_paths, df_data = self._phenoscorer.load_data_from_excel(os.path.join("..", "sample_data",
                                                                                       "satb1_data.xlsx"))
        self._phenoscorer.get_lime(X, y, img_paths, n_lime=5)
        self._phenoscorer.gen_lime_and_results_figure(bg_image=os.path.join("..", "sample_data",
                                                                            "background_image.jpg"),
                                                      df_data=df_data, filename='lime_figure_' + 'SATB1_test.pdf')
