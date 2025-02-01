import unittest
import numpy as np
from random import shuffle
from phenoscore.models.svm import get_loss
from phenoscore.hpo_phenotype.calc_hpo_sim import SimScorer
from phenoscore.phenoscorer import PhenoScorer
from sklearn.metrics import brier_score_loss
import os


class SVMTester(unittest.TestCase):
    def setUp(self):
        try:
            self._simscorer = SimScorer(
                similarity_data_path=os.path.join('phenoscore', 'hpo_phenotype'),
                hpo_network_csv_path=os.path.join('phenoscore', 'hpo_phenotype', 'hpo_network.csv'),
                name_to_id_json=os.path.join('phenoscore', 'hpo_phenotype', 'hpo_name_to_id.json')
            )
        except:
            self._simscorer = SimScorer(
                similarity_data_path=os.path.join('..', 'hpo_phenotype'),
                hpo_network_csv_path=os.path.join('..', 'hpo_phenotype', 'hpo_network.csv'),
                name_to_id_json=os.path.join('..', 'hpo_phenotype', 'hpo_name_to_id.json')
            )
        self._phenoscorer = PhenoScorer(gene_name='random', mode='both')

    def test_svm(self):
        nodes = list(self._simscorer.hpo_network.nodes())
        for N_PATIENTS in [3, 4, 5, 6, 8, 12, 22]:
            hpo_scores, face_scores, both_scores = [], [], []
            for z in range(5):
                y = np.array([0, 1] * int(N_PATIENTS))
                y = np.random.permutation(y)

                X = np.zeros((int(N_PATIENTS*2), 2623), dtype=object)
                for i in range(len(X)):
                    face_rand = np.random.uniform(-1, 1, 2622)
                    hpo_rand = list(np.random.choice(nodes, size=np.random.randint(3, 30), replace=False))

                    X[i, :2622] = face_rand
                    X[i, 2622] = hpo_rand

                sim_mat = self._simscorer.calc_full_sim_mat(X)
                y_real, y_pred, y_ind = get_loss(X, y, self._simscorer, 'hpo', sim_mat)
                hpo_scores.append(brier_score_loss(y_real, y_pred))
                y_real, y_pred, y_ind = get_loss(X, y, self._simscorer, 'face', sim_mat)
                face_scores.append(brier_score_loss(y_real, y_pred))
                y_real, y_pred, y_ind = get_loss(X, y, self._simscorer, 'both', sim_mat)
                both_scores.append(brier_score_loss(y_real, y_pred))

            print("Patients: " + str(N_PATIENTS) + ' mean Brier HPO:' + str(hpo_scores))
            print("Patients: " + str(N_PATIENTS) + ' mean Brier Face:' + str(face_scores))
            print("Patients: " + str(N_PATIENTS) + ' mean Brier Both:' + str(both_scores))

            assert np.mean(hpo_scores) > 0.2
            assert np.mean(face_scores) > 0.2
            assert np.mean(both_scores) > 0.2

