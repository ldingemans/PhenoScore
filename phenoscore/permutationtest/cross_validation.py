import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import pandas as pd
from phenoscore.models.svm import get_loss, svm_class
from phenoscore.explainability_lime.LIME import explain_prediction


class CrossValidatorAndLIME:
    def __init__(self, n_lime: int) -> None:
        """
        Constructor

        Parameters
        ----------
        n_lime: int
            Number of top predictions to generate LIME predictions for
        """
        self._n_lime = n_lime
        self.results = None

    def get_results(self, x: np.ndarray, y: np.ndarray, img_paths: np.ndarray, mode: str, simscorer, facial_feature_extractor) -> 'CrossValidatorAndLIME':

        """
        Small wrapper for function below that gets cross-validation results and LIME explanations.

        Parameters
        ----------
        X: numpy array
            Array of size n x 2623: the VGG-Face feature vector and one cell with a list of the HPO IDs
        y: numpy array
            The labels (usually 0 for control and 1 for patient)
        img_paths: numpy array
            One dimensional array with the file paths to the images of patients and controls.
            These are needed for the LIME explanations, since we need to pertube the original images then.
        mode: str
            Whether to use facial data, HPO terms, or both
        simscorer: SimScorer object
            Instance of the SimScorer class of this package
        facial_feature_extractor: FacialFeatureExtractor object
            Instance to extract facial features, default is VGGFace, can be QMagFace as well
        """
        self.results = self._get_results_loo_cv(x, y, img_paths, mode, simscorer, facial_feature_extractor)
        return self

    def _get_results_loo_cv(self, x: np.ndarray, y: np.ndarray, file_paths: np.ndarray, mode: str, simscorer, facial_feature_extractor) -> pd.DataFrame:
        """
        Get the results after cross-validation of the classifier for a genetic syndrome.
        Generate LIME explanations for the test predictions as well. These are the main analyses of the paper.

        Parameters
        ----------
        X: numpy array
            Array of size n x 2623: the VGG-Face feature vector and one cell with a list of the HPO IDs
        y: numpy array
            The labels (usually 0 for control and 1 for patient)
        file_paths: numpy array
            One dimensional array with the file paths to the images of patients and controls. These are needed for the LIME explanations, since we need to pertube the original images then
        mode: str
            PhenoScore mode, either hpo/face/both depending on the data available and therefore which analysis to run.
        simscorer: object of SimScorer class
            Instance of class for semantic similarity calculations
        facial_feature_extractor: FacialFeatureExtractor object
            Instance to extract facial features, default is VGGFace, can be QMagFace as well

        Returns
        -------
        df_results: pandas DataFrame
            DataFrame with the results
        """
        assert ((mode == 'both') or (mode == 'face') or (mode == 'hpo'))

        y_pred_all_svm = np.zeros((len(x), 3))

        if mode != 'face':
            sim_mat = simscorer.calc_full_sim_mat(x)

        print("Starting cross validation procedure to compare using facial/HPO data only with PhenoScore.")

        if mode != 'hpo':
            y_pred_indexer = 0
            y_real_test, y_pred_all_svm[:, y_pred_indexer], y_test_ind = get_loss(x[:, :facial_feature_extractor.
                                                                                  face_vector_size], y, simscorer,
                                                                                  mode='face', sim_mat=None)
        if mode != 'face':
            y_pred_indexer = 1
            y_real_test, y_pred_all_svm[:, y_pred_indexer], y_test_ind = get_loss(x, y, simscorer,
                                                                                  mode='hpo', sim_mat=sim_mat)
        if mode == 'both':
            y_pred_indexer = 2
            y_real_test, y_pred_all_svm[:, y_pred_indexer], y_test_ind = get_loss(x, y, simscorer,
                                                                                  mode='both', sim_mat=sim_mat)

        print("Finished cross validation and evaluation of model scores. Now starting LIME for the top " + str(
            self._n_lime) + " predictions to generate heatmaps and visualise phenotypic differences.")
        # get the LIME predictions for the top n_top, default is 5: can do for all, but takes a long time (especially facial images)

        explanations_face, explanations_hpo, local_preds_face, local_preds_hpo, file_paths_sorted = [], [], [], [], []

        phenoscore = pd.Series(y_pred_all_svm[:, y_pred_indexer])
        highest_predictions = phenoscore[np.array(y_real_test) == 1].nlargest(self._n_lime)
        highest_predictions = y_test_ind[np.array(highest_predictions.index)]

        pbar = tqdm(total=self._n_lime)

        for z in y_test_ind:
            if mode == 'face' or mode == 'both':
                file_paths_sorted.append(file_paths[z])

            if z in highest_predictions:
                assert (y[z] == 1)
                # we need to retrain the model to get clf and other variables with this instance in the test set and rest as training data
                test_index = np.array([z])
                all_indices = np.array(range(len(y)))
                train_index = all_indices[all_indices != test_index]

                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if mode != 'face':
                    sim_mat_train, sim_mat_test = simscorer.calc_sim_scores(sim_mat, train_index, test_index,
                                                                                  y_train)
                    hpo_terms_pt, hpo_terms_cont = X_train[y_train == 1, -1].reshape(-1, 1), X_train[
                        y_train == 0, -1].reshape(-1, 1)
                    scale_hpo = StandardScaler()
                    X_hpo_train_norm = scale_hpo.fit_transform(sim_mat_train)
                    X_hpo_test_norm = scale_hpo.transform(sim_mat_test)

                if mode != 'hpo':
                    X_face_train = np.array(X_train[:, :-1], dtype=float)
                    X_face_test = np.array(X_test[:, :-1], dtype=float)

                    scale_face = StandardScaler()
                    X_face_train_norm = normalize(scale_face.fit_transform(X_face_train))
                    X_face_test_norm = normalize(scale_face.transform(X_face_test))

                if mode == 'both':
                    X_train = np.append(X_face_train_norm, X_hpo_train_norm, axis=1)
                    X_test = np.append(X_face_test_norm, X_hpo_test_norm, axis=1)

                if mode == 'hpo':
                    preds_svm_hpo, clf = svm_class(X_hpo_train_norm, y_train, X_hpo_test_norm)
                elif mode == 'face':
                    preds_svm_face, clf = svm_class(X_face_train_norm, y_train, X_face_test_norm)
                elif mode == 'both':
                    preds_svm_both, clf = svm_class(X_train, y_train, X_test)

                if mode == 'hpo':
                    exp_face, local_pred_face, exp_hpo, local_pred_hpo, csi, vsi, rsquared, kernel_width, sum_coeffs = explain_prediction(x, z, clf, None, scale_hpo,
                                                                                            hpo_terms_pt,
                                                                                            hpo_terms_cont,
                                                                                            simscorer,
                                                                                            simscorer.name_to_id_and_reverse
                                                                                            )
                elif mode == 'both':
                    exp_face, local_pred_face, exp_hpo, local_pred_hpo, csi, vsi, rsquared, kernel_width, sum_coeffs = explain_prediction(x, z, clf, scale_face,
                                                                                            scale_hpo, hpo_terms_pt,
                                                                                            hpo_terms_cont,
                                                                                            simscorer,
                                                                                            simscorer.name_to_id_and_reverse,
                                                                                            str(file_paths[z]),
                                                                                            n_iter=100,
                                                                                            facial_feature_extractor=facial_feature_extractor)
                elif mode == 'face':
                    exp_face, local_pred_face, exp_hpo, local_pred_hpo, csi, vsi, rsquared, kernel_width, sum_coeffs  = explain_prediction(x, z, clf, scale_face,
                                                                                            img_path_index_patient=str(
                                                                                                file_paths[z]),
                                                                                            n_iter=100,
                                                                                            facial_feature_extractor=facial_feature_extractor)

                explanations_face.append(exp_face)
                explanations_hpo.append(exp_hpo)
                local_preds_face.append(local_pred_face)
                local_preds_hpo.append(local_pred_hpo)
                pbar.update(1)
            else:
                explanations_face.append('')
                explanations_hpo.append('')
                local_preds_face.append('')
                local_preds_hpo.append('')

        pbar.close()
        results = np.ones(17, dtype=object)
        # Calculate metrics and store them in the results list
        results[0] = roc_auc_score(y_real_test, y_pred_all_svm[:, 0])
        results[1] = brier_score_loss(y_real_test, y_pred_all_svm[:, 0])
        results[2] = roc_auc_score(y_real_test, y_pred_all_svm[:, 1])
        results[3] = brier_score_loss(y_real_test, y_pred_all_svm[:, 1])
        results[4] = roc_auc_score(y_real_test, y_pred_all_svm[:, 2])
        results[5] = brier_score_loss(y_real_test, y_pred_all_svm[:, 2])

        # Store explanations and predictions
        results[6] = explanations_face
        results[7] = explanations_hpo
        results[8] = local_preds_face
        results[9] = local_preds_hpo
        results[10] = y_pred_all_svm[:,y_pred_indexer]
        results[11] = y_real_test

        # Store additional information
        results[12] = file_paths_sorted
        results[13] = csi
        results[14] = vsi
        results[15] = rsquared
        results[16] = local_pred_hpo
        df_results = pd.DataFrame(results).T
        df_results.columns = [
    'roc_face_svm', 'brier_face_svm', 'roc_hpo_svm', 'brier_hpo_svm', 'roc_both_svm',
    'brier_both_svm', 'face_explanation', 'hpo_explanation', 'face_pred', 'hpo_pred',
    'svm_preds', 'y_real', 'file_paths', 'CSI', 'VSI', 'R^2', 'local_pred_hpo'
]
        return df_results
