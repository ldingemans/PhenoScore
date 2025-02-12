import sys
from phenoscore.models.svm import get_loss
from scipy import stats
from scipy.stats import mannwhitneyu
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from tqdm import tqdm

sys.setrecursionlimit(1500)


class PermutationTester:
    def __init__(self, simscorer, mode='both', bootstraps=100):
        """Constructor
        simscorer: object of SimScorer class
            Instance of class for semantic similarity calculations
        mode: str
            Whether to use facial data, HPO terms, or both
        bootstraps: int
            Number of times to do the bootstrapping procedure
        """
        assert ((mode == 'both') or (mode == 'face') or (mode == 'hpo'))
        self._mode = mode
        self._bootstraps = bootstraps
        self._simscorer = simscorer
        self.classifier_results = None
        self.bootstrapped_results = None
        self.ps = None
        self.p_value = None
        self.classifier_aucs = None

    def _c2st(self, X, y, pbar=None):
        """
        Perform Classifier Two Sample Test (C2ST) by randomizing the y labels, to obtain a p-value for the classification results.
        Inspired by Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.

        Parameters
        ----------
        X: numpy array
            Array of size n x 2623 of the original patients and controls of the suspected syndrome: the
            VGG-Face2 feature vector and one cell with a list of the HPO IDs
        y: numpy array
            The y labels
        pbar: tqdm instance
            Progressbar to fill

        Returns
        -------
        emp_loss: float
            Brier score of the real classifier
        bs_losses: numpy array
            The Brier scores of the classifiers trained on the shuffled y labels
        emp_loss_auc: float
            AROC of the real classifier
        """
        bs_losses = []

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._mode != 'face':
            sim_mat = self._simscorer.calc_full_sim_mat(X)
        else:
            sim_mat = None

        y_real, y_pred, y_ind = get_loss(
            X, y, self._simscorer, self._mode, sim_mat)
        emp_loss, emp_loss_auc = brier_score_loss(
            y_real, y_pred), roc_auc_score(y_real, y_pred)
        pbar.update(1)
        for b in range(self._bootstraps):
            y_random = np.random.permutation(y)
            y_real, y_pred, y_ind = get_loss(
                X, y_random, self._simscorer, self._mode, sim_mat)
            bs_losses.append(brier_score_loss(y_real, y_pred))
            pbar.update(1)
        return emp_loss, np.array(bs_losses), emp_loss_auc

    def permutation_test_multiple_X(self, X, y):
        """
        Do the permutation test for every entry in X and y and obtain a p-value for the classifier
        when having resampled controls multiple times.

        Parameters
        ----------
        X: list
            List of arrays of size n x 2623 of the original patients and controls of the suspected syndrome: the
            VGG-Face2 feature vector and one cell with a list of the HPO IDs.
            This can be used if we are resampling from a control database for instance multiple times.
        y: list
            The y labels

        Returns
        -------
        classifier_results: list
            Brier scores of the classifier on the real/unpermuted data
        bootstrapped_results: list
            Brier scores of the classifier on the permuted data
        ps: list
            P-values obtained during the resampling
        p_value: float
            Obtained final p-value
        classifier_aucs: list
            AROC scores of the classifier on the real/unpermuted data
        """

        bootstrapped_results = []
        classifier_results = []
        classifier_aucs = []
        ps = []

        pbar = tqdm(total=len(X) * self._bootstraps + len(X))

        for z in range(len(X)):
            acc, random_losses, auc = self._c2st(X[z], y[z], pbar)
            if np.mean(y[z]) != 0.5:
                print(
                    "WARNING: the dataset is imbalanced. This permutation test has not been validated for imbalanced "
                    "datasets, it is therefore recommended to undersample the majority class. "
                    "The test will however continue now.")

            classifier_results.append(acc)
            classifier_aucs.append(auc)
            bootstrapped_results.extend(random_losses)
            ps.append(mannwhitneyu(acc, random_losses,
                      alternative='less', nan_policy='raise')[1])

        p_value = stats.combine_pvalues(ps, method='fisher', weights=None)[1]

        assert len(bootstrapped_results) == (self._bootstraps * len(X))
        assert len(classifier_results) == len(X)

        self.classifier_results = classifier_results
        self.bootstrapped_results = bootstrapped_results
        self.ps = ps
        self.p_value = p_value
        self.classifier_aucs = classifier_aucs
        return self

    def permutation_test(self, X, y):
        """
        Do the permutation test for X and y and obtain a p-value for the classifier.

        Parameters
        ----------
        X: numpy array
            Single arrray of size n x 2623 of the original patients and controls of the suspected syndrome: the
            VGG-Face2 feature vector and one cell with a list of the HPO IDs.
        y: numpy array
            The y labels

        Returns
        -------
        classifier_results: list
            Brier scores of the classifier on the real/unpermuted data
        bootstrapped_results: list
            Brier scores of the classifier on the permuted data
        ps: list
            P-values obtained during the resampling
        p_value: float
            Obtained final p-value
        classifier_aucs: list
            AROC scores of the classifier on the real/unpermuted data
        """
        if np.mean(y) != 0.5:
            print(
                "WARNING: the dataset is imbalanced. This permutation test has not been validated for imbalanced "
                "datasets, it is therefore recommended to undersample the majority class. "
                "The test will however continue now.")

        pbar = tqdm(total=self._bootstraps + 1)

        classifier_results, bootstrapped_results, classifier_aucs = self._c2st(
            X, y, pbar)
        p_value = mannwhitneyu(
            classifier_results, bootstrapped_results, alternative='less', nan_policy='raise')[1]

        self.classifier_results = classifier_results
        self.bootstrapped_results = bootstrapped_results
        self.p_value = p_value
        self.classifier_aucs = classifier_aucs
        return self
