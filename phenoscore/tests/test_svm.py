import unittest
import numpy as np
from random import shuffle
from phenoscore.models.svm import get_loss
from phenoscore.hpo_phenotype.calc_hpo_sim import SimScorer
from phenoscore.phenoscorer import PhenoScorer
from sklearn.metrics import brier_score_loss


class SVMTester(unittest.TestCase):
    def setUp(self):
        self._simscorer = SimScorer(scoring_method='Resnik', sum_method='BMA')
        self._phenoscorer = PhenoScorer(gene_name='random', mode='both')

    def test_svm(self):
        nodes = list(self._simscorer.hpo_network.nodes())
        for N_PATIENTS in [8, 12, 22]:
            y = np.array([0, 1] * int(N_PATIENTS / 2))
            y = np.random.permutation(y)

            X = np.zeros((N_PATIENTS, 2623), dtype=object)
            for i in range(len(X)):
                face_rand = np.random.uniform(-1, 1, 2622)
                hpo_rand = list(np.random.choice(nodes, size=np.random.randint(3, 30), replace=False))

                X[i, :2622] = face_rand
                X[i, 2622] = hpo_rand

            sim_mat = self._simscorer.calc_full_sim_mat(X)
            y_real, y_pred, y_ind = get_loss(X, y, self._simscorer, 'hpo', sim_mat)
            assert brier_score_loss(y_real, y_pred) > 0.2
            y_real, y_pred, y_ind = get_loss(X, y, self._simscorer, 'face', sim_mat)
            assert brier_score_loss(y_real, y_pred) > 0.2
            y_real, y_pred, y_ind = get_loss(X, y, self._simscorer, 'both', sim_mat)
            assert brier_score_loss(y_real, y_pred) > 0.2