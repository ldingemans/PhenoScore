import unittest
from phenoscore.phenoscorer import PhenoScorer
import os


class CrossValLIMETester(unittest.TestCase):
    def setUp(self):
        self._phenoscorer = PhenoScorer(gene_name='SATB1', mode='both')

    def test_lime_gen(self):
        path_to_script = os.path.realpath(__file__).split(os.sep)[:-2]
        path_to_script.insert(1, os.sep)
        X, y, img_paths, df_data = self._phenoscorer.load_data_from_excel(os.path.join(*path_to_script, "sample_data",
                                                                                   "satb1_data.xlsx"))
        self._phenoscorer.get_lime(X, y, img_paths, n_lime=1)
        assert self._phenoscorer.lime_results is not None
