import ast
import os
from typing import List, Tuple, Optional
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import normalize


from phenoscore.explainability_lime.LIME import explain_prediction
from phenoscore.facial_feature_extraction.extract_facial_features import VGGFaceExtractor, QMagFaceExtractor
from phenoscore.hpo_phenotype.calc_hpo_sim import SimScorer
from phenoscore.models.svm import get_clf
from phenoscore.permutationtest.cross_validation import CrossValidatorAndLIME
from phenoscore.permutationtest.permutation_test import PermutationTester
from phenoscore.tables_and_figures.gen_tables_and_figs import get_top_hpo, get_heatmap_from_multiple
from phenoscore.explainability_lime.LIME import LIMEConfiguration, OptiLIMEConfiguration
os.environ["MXNET_SUBGRAPH_VERBOSE"] = "0"

class PhenoScorer:
    """ Main class used to train, test, generate, and explain predictions
        for the PhenoScore model based on facial features, HPO terms, or both."""
    def __init__(
    self,
    gene_name: str,
    mode: str,
    method_hpo_similarity='Resnik',
    method_summ_hpo_similarities='BMA',
    face_module='VGGFace'
) -> None:
        """
        Constructor

        Parameters
        ----------
        gene_name: str
            Name of gene for the LIME plot
        mode: str
            Whether to use facial data, HPO terms, or both
        method_hpo_similarity: str
            Scoring method to use to calculate semantic similarity.
            Can be HRSS, Resnik, Jaccard or word2vec
        method_summ_hpo_similarities: str
            Method to summarize the HPO term similarities.
            Can be BMA, BMWA, or maximum.
        face_module: str
            Method to extract facial features, default is VGGFace, can be QMagFace as well.
        """
        devices = tf.config.list_physical_devices('GPU')
        if len(devices) == 0:
            print('Using CPU, since no GPUs are found!')
        else:
            print(f'Using GPUs: {devices}')

        assert mode in ('both', 'face', 'hpo')
        self.gene_name = gene_name
        self.mode = mode
        self.permutation_test_brier = None
        self.permutation_test_auc = None
        self.permutation_test_p_value = None
        self.lime_results = None

        self._simscorer = SimScorer(scoring_method=method_hpo_similarity,
                                    sum_method=method_summ_hpo_similarities)
        if mode in ('both', 'face'):
            if face_module == 'VGGFace':
                self._facial_feature_extractor = VGGFaceExtractor()
            elif face_module == 'QMagFace':
                path_to_script = os.path.realpath(__file__).split(os.sep)[:-1]
                path_to_script.insert(1, os.sep)
                path_to_qmagface = os.path.join(*path_to_script, 'facial_feature_extraction')
                self._facial_feature_extractor = QMagFaceExtractor(path_to_dir=path_to_qmagface)
            else:
                raise ValueError('Invalid facial recognition module chosen')
        else:
            self._facial_feature_extractor = None
        self.vus_results = None

    
    def load_and_process_data_from_excel(
    self, path_to_excel_file: str
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """
        Load an excel file with the same columns as the provided examples, such as the
        random_generated_sample_data.xlsx

        Parameters
        ----------
        path_to_excel_file: str
            Path to excel file to process

        Returns
        -------
        X: numpy array
            If facial photos included: Array of size n x 2623 of the original patients
            and controls of the suspected syndrome: the VGG-Face2 feature vector and
            one cell with a list of the HPO IDs. Else: Array of size n x 1 with the HPO IDs.
        y: numpy array
            The true class labels of the data where 1 denotes a patient and 0 a control.
        img_paths: numpy array
            One dimensional array with the file paths to the images of patients and controls.
            Needed for LIME explanations, since we need to perturb the original images.
        df_data: pandas DataFrame
            DataFrame containing the HPO terms, HPO terms including parents, and the graph
            of the HPO terms and/or the facial feature vectors.
        """
        df_data = pd.read_excel(path_to_excel_file) if ".xlsx" in path_to_excel_file else path_to_excel_file

        if self.mode in ['hpo', 'both']:
            df_data = self.filter_parent_hpo_terms(df_data)

        df_data['graph'], df_data['hpo_name_inc_parents'] = '', ''  

        if (self.mode == 'both') or (self.mode == 'face'):
            x = np.zeros((len(df_data), self._facial_feature_extractor.face_vector_size + 1), dtype=object)
        elif self.mode == 'hpo':
            x = np.zeros((len(df_data), 1), dtype=object)
        y = df_data.loc[:, 'y_label'].to_numpy()
        img_paths = []
        # convert images to VGG-Face feature vector and create the appropriate numpy array x
        for i in range(len(df_data)):
            if self.mode in ['both', 'face']:
                full_img_path = os.path.join(os.path.dirname(path_to_excel_file), df_data.loc[i, 'path_to_file'].replace('\\', os.sep))
                x[i, :self._facial_feature_extractor.face_vector_size] = self._facial_feature_extractor.process_file(full_img_path)
                img_paths.append(full_img_path)
            if self.mode in ['hpo', 'both']:
                x, df_data = self.process_hpo_data(df_data, i, x, 0 if self.mode == 'hpo' else self._facial_feature_extractor.face_vector_size)

        if self.mode in ['both', 'face']:
            x, y, img_paths, df_data = self.handle_unprocessed_photos(x, y, img_paths, df_data)

        return x, y, img_paths, df_data


    def filter_parent_hpo_terms(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out parent HPO terms from the 'hpo_all' column in the provided DataFrame.

        This method iterates over each row in the DataFrame, evaluates the 'hpo_all' column
        as a list of HPO terms, removes the root term 'HP:0000001' to avoid bugs in subsequent
        processing, and then uses a SimScorer instance to remove parent HPO terms, leaving
        a more concise set of HPO terms that still accurately represent the phenotypic features.

        Parameters
        ----------
        df_data : pd.DataFrame
            The DataFrame containing HPO terms. It must have a column named 'hpo_all'
            which contains string representations of lists of HPO terms.

        Returns
        -------
        pd.DataFrame
            The modified DataFrame with parent HPO terms filtered out from the 'hpo_all' column.
        """
        simscorer = SimScorer()
        for i in range(len(df_data)):
            all_terms = ast.literal_eval(df_data.loc[i,'hpo_all'])
            all_terms = [term for term in all_terms if term != 'HP:0000001']
            without_parents = simscorer.delete_parent_terms(all_terms)
            df_data.loc[i,'hpo_all'] = str(without_parents)
        return df_data

    def process_hpo_data(self, df_data: pd.DataFrame, i: int, x: np.ndarray, index: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Processes HPO data for a single row in the DataFrame by updating an array and the DataFrame.

        This method takes HPO terms from a specified row in the DataFrame, evaluates them, and updates
        a given index in the numpy array `X` with these terms. It also updates the DataFrame with a graph
        representation of these HPO terms and a list of HPO terms including their parents.

        Parameters
        ----------
        df_data : pd.DataFrame
            The DataFrame containing HPO terms in the 'hpo_all' column.
        i : int
            The index of the row in `df_data` to process.
        x : np.ndarray
            The numpy array to update with HPO terms from `df_data`.
        index : int
            The index in `x` where the HPO terms should be updated.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            A tuple containing the updated numpy array and DataFrame.
        """
        hpo_all = ast.literal_eval(df_data.loc[i, 'hpo_all'])
        x[i, index] = hpo_all
        df_data.at[i, 'graph'] = self._simscorer.get_graph(hpo_all, False)
        df_data.at[i, 'hpo_name_inc_parents'] = list(df_data.at[i, 'graph'].nodes())
        return x, df_data

    def handle_unprocessed_photos(self, x: np.ndarray, y: np.ndarray, 
                                img_paths: np.array, df_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.array, pd.DataFrame]:
        """
    Filters out entries corresponding to photographs where no face was detected.

    This method identifies and removes rows across the provided numpy arrays and DataFrame
    where the facial photograph processing failed (indicated by NaN values in the first column
    of `x`). It also prints the paths of the photographs that failed to process.

    Parameters
    ----------
    x : np.ndarray
        A numpy array containing processed facial photograph data.
    y : np.ndarray
        A numpy array containing the class labels corresponding to each photograph.
    img_paths : np.array
        A numpy array containing the file paths to each photograph.
    df_data : pd.DataFrame
        A DataFrame containing additional data related to each photograph.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.array, pd.DataFrame]
        A tuple containing the updated versions of `x`, `y`, `img_paths`, and `df_data`
        with entries corresponding to unprocessed photographs removed.
    """

        indices_not_processed_photographs = np.isnan(x[:, 0].astype(float))
        if np.sum(indices_not_processed_photographs) > 0:
            print('The following facial photographs failed to process, '
                'because no face was detected:')
            for i, _ in enumerate(indices_not_processed_photographs):
                if indices_not_processed_photographs[i]:
                    print(img_paths[i])
            x = x[~indices_not_processed_photographs]
            y = y[~indices_not_processed_photographs]
            img_paths = list(np.array(img_paths)[~indices_not_processed_photographs])
            df_data = df_data.loc[~indices_not_processed_photographs, :].reset_index(drop=True)
        return x, y, img_paths, df_data

    def permutation_test(self, x: np.ndarray, y: np.ndarray, bootstraps=1000) -> 'PhenoScorer':
        """

        Parameters
        ----------
        x: numpy array
            Array of size n x 2623 of the original patients and controls of the suspected
            syndrome: the VGG-Face2 feature vector and one cell with a list of the HPO IDs.
        y: numpy array
            The y labels of the data.
        bootstraps: int
            Number of bootstraps for permutation test.
        """
        permutation_tester = PermutationTester(self._simscorer, mode=self.mode, bootstraps=bootstraps)
        if isinstance(x, list): #Analysis with multiple sets of controls.
            permutation_tester.permutation_test_multiple_X(x, y)
        else:
            permutation_tester.permutation_test(x, y)
        self.permutation_test_brier = np.mean(permutation_tester.classifier_results)
        self.permutation_test_auc = np.mean(permutation_tester.classifier_aucs)
        self.permutation_test_p_value = np.mean(permutation_tester.p_value)
        return self

    def get_lime(self, x: np.ndarray, y: np.ndarray, img_paths: np.ndarray, n_lime=5) -> 'PhenoScorer':
        """

        Parameters
        ----------
        x: numpy array
            Array of size n x 2623 of the original patients and controls of the suspected
            syndrome: the VGG-Face2 feature vector and one cell with a list of the HPO IDs.
        y: numpy array
            The y labels of the data.
        img_paths: numpy array
            One dimensional array with the file paths to the images of patients and controls.
            These are needed for the LIME explanations, since we need to pertube the original images the
        n_lime: int
            Number of top predictions to generate LIME predictions for
        """
        cross_val_and_limer = CrossValidatorAndLIME(n_lime=n_lime)
        cross_val_and_limer.get_results(x, y, img_paths, self.mode, self._simscorer, self._facial_feature_extractor)
        self.lime_results = cross_val_and_limer.results
        return self

    def gen_lime_and_results_figure(self, bg_image: str, df_data: pd.DataFrame, filename: str, bubble_plot=False, lime_iterations=1) -> None:
        """
        Plot the LIME and results figure of the paper

        Parameters
        ----------
        bg_image: str
            Path to image to use as background for the facial heatmap
        df_data: pandas DataFrame
            Original dataframe with raw data, used to plot the original prevalences of the HPO terms in the figure to
            check whether the recovered HPO terms make sense.
        filename: str
            Filename of exported figure
        bubble_plot: bool
            Whether to make plot as in the figure of the paper or plot regular bars
        lime_iterations: int
            If your lime results are actually result of multiple iterations, specify the number here.
            It is only used to correct the n in the figures by dividing by the number of iterations.
        """
        if self.lime_results is None:
            raise ValueError("Please run get_lime first to obtain LIME results.")

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        if self.mode == 'face' or self.mode == 'both':
            mean_face_norm = self._facial_feature_extractor.get_norm_image(bg_image)

        score_aroc_both = np.round(self.lime_results.loc[0, 'roc_both_svm'], 2)
        score_aroc_face = np.round(self.lime_results.loc[0, 'roc_face_svm'], 2)
        score_aroc_hpo = np.round(self.lime_results.loc[0, 'roc_hpo_svm'], 2)
        if self.mode == 'both':
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 1, figsize=(6, 5))
            axs = np.array([axs])

        df_exp = pd.DataFrame()
        df_exp['hpo_pred'] = np.array(self.lime_results.loc[0, 'hpo_pred'])
        df_exp['face_pred'] = np.array(self.lime_results.loc[0, 'face_pred'])
        for i in range(len(df_exp)):
            if type(df_exp.loc[i, 'face_pred']) == list:
                df_exp.loc[i, 'face_pred'] = np.mean(df_exp.loc[i, 'face_pred'])
        df_exp['svm_pred'] = np.array(self.lime_results.loc[0, 'svm_preds'])
        df_exp['hpo_exp'] = np.array(self.lime_results.loc[0, 'hpo_explanation'])
        df_exp['face_exp'] = np.array(self.lime_results.loc[0, 'face_explanation'])
        df_exp['y_true'] = np.array(self.lime_results.loc[0, 'y_real'])

        if self.mode == 'face' or self.mode == 'both':
            df_top = df_exp[df_exp['face_exp'] != ''].reset_index(drop=True)
        else:
            df_top = df_exp[df_exp['hpo_exp'] != ''].reset_index(drop=True)

        if self.mode == 'hpo' or self.mode == 'both':
            df_summ_hpo = get_top_hpo(df_top.loc[:, 'hpo_exp'].explode(), True)

        if self.mode == 'face' or self.mode == 'both':
            fig = get_heatmap_from_multiple(df_top.loc[:, 'face_exp'].explode(), fig, axs[0], mean_face_norm, 0.5, self._facial_feature_extractor.input_image_size)
            axs[0].set_title('Face: ' + str(score_aroc_face), fontsize=18, fontweight='bold')
            axs[0].set_ylabel(self.gene_name, fontsize=18, fontweight='bold', style='italic')
            x_n_text = int(self._facial_feature_extractor.input_image_size[0] / 2.4)
            y_n_text = int(self._facial_feature_extractor.input_image_size[0] * 1.1)
            axs[0].annotate("n=" + str(int(np.sum(self.lime_results.loc[0, 'y_real'] == 1)/lime_iterations)),
                            (x_n_text, y_n_text),
                            annotation_clip=False,
                            fontsize=16, fontweight='bold')

        if self.mode == 'hpo' or self.mode == 'both':
            axs[len(axs) - 1].set_title('HPO: ' + str(score_aroc_hpo), fontsize=18, fontweight='bold')

            df_summ_hpo = df_summ_hpo.sort_values('corr', ascending=False)
            if bubble_plot == True:
                df_summ_hpo = df_summ_hpo.loc[df_summ_hpo['corr'] > 0, :] # only use positive correlations for this plot
                x = [0] * len(df_summ_hpo)
                y = np.flip(range(len(df_summ_hpo)))
                axs[len(axs) - 1].scatter(x, y, alpha=0.5, s=df_summ_hpo['corr'].to_numpy() * 30000)
                for i, txt in enumerate(list(df_summ_hpo.index)):
                    txt = txt.replace(' greater than ', ' > ')
                    txt = txt.replace(' less than ', ' < ')
                    if txt == 'Febrile seizure (within the age range of 3 months to 6 years)':
                        txt = 'Febrile seizure (within the age\nrange of 3 months to 6 years)'
                    if txt == 'Delayed speech and language development':
                        txt = 'Delayed speech and\n language development'
                    if txt == 'Birth length > 97th percentile':
                        txt = 'Birth length >\n 97th percentile'
                    if txt == 'Gastrostomy tube feeding in infancy':
                        txt = 'Gastrostomy tube feeding\nin infancy'
                    axs[len(axs) - 1].annotate(txt, (x[i] + 0.12, y[i]), verticalalignment="center",
                                           horizontalalignment="left", fontsize=16, fontweight='semibold')
                axs[len(axs) - 1].set_xlim(-0.2, 0.5)
                axs[len(axs) - 1].set_ylim(-2, 6)
                axs[len(axs) - 1].set_axis_off()
            else:
                for hpo_term in df_summ_hpo['hpo']:
                    df_summ_hpo.loc[hpo_term, 'prev_0'] = df_data.loc[
                        df_data['y_label'] == 0, 'hpo_name_inc_parents'].astype(str).str.contains(hpo_term).mean()
                    df_summ_hpo.loc[hpo_term, 'prev_1'] = df_data.loc[
                        df_data['y_label'] == 1, 'hpo_name_inc_parents'].astype(str).str.contains(hpo_term).mean()

                g = sns.barplot(x=df_summ_hpo['corr'], y=list(df_summ_hpo.index), color='blue', alpha=0.6, ax=axs[len(axs) - 1])

                for bar in g.patches:
                    if bar.get_width() < 0:
                        bar.set_color('red')
                axs[len(axs) - 1].set_xlim(-0.25, 0.25)
                axs[len(axs) - 1].set_yticks([])
                axs[len(axs) - 1].axes.yaxis.set_visible(False)
                axs[len(axs) - 1].spines['left'].set_visible(False)
                axs[len(axs) - 1].spines['right'].set_visible(False)
                axs[len(axs) - 1].spines['top'].set_visible(False)
                axs[len(axs) - 1].set_xlabel('LIME regression coefficient')
                df_summ_hpo = df_summ_hpo.reset_index(drop=True)
                for y in range(len(df_summ_hpo)):
                    axs[len(axs) - 1].text(0, y, df_summ_hpo.loc[y, 'hpo'], fontsize=12, horizontalalignment='center',
                                verticalalignment='center', fontweight='semibold')
                    axs[len(axs) - 1].text(-0.28, y, str(int(np.round(df_summ_hpo.loc[y, 'prev_0'] * 100))) + '%', fontsize=12,
                                horizontalalignment='left', verticalalignment='center')
                    axs[len(axs) - 1].text(0.28, y, str(int(np.round(df_summ_hpo.loc[y, 'prev_1'] * 100))) + '%', fontsize=12,
                                horizontalalignment='right', verticalalignment='center')
                if self.lime_results.loc[0,'CSI'] > 0:
                    axs[-1].text(-0.25, -2, f'''CSI = {self.lime_results['CSI'].iloc[0]}%''', fontsize=12, horizontalalignment='left')
                    axs[-1].text(0, -2, f'''VSI = {self.lime_results['VSI'].iloc[0]}%''', fontsize=12, horizontalalignment='center')
                if self.lime_results.loc[0,'R^2'] > 0:
                    axs[-1].text(0.25, -2, rf'''$R^2={round(self.lime_results['R^2'].iloc[0], 1)}$''', fontsize=12, horizontalalignment='right')
        if self.mode == 'both':
            fig.suptitle('PhenoScore: ' + str(score_aroc_both), fontsize=20, fontweight='bold', horizontalalignment='center', x=0.53)
        plt.subplots_adjust(top=0.77)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print("Figure saved as " + filename)
        plt.show()
        return

    def predict_new_sample(self, original_x: np.ndarray, original_y: np.ndarray, img: str, hpo_all_new_sample: List, LIME_config: LIMEConfiguration = LIMEConfiguration(), OptiLIME_config: OptiLIMEConfiguration = OptiLIMEConfiguration(), lime_iter=100, kernel_width=None) -> 'PhenoScorer':
        """
        Train a classifier, get prediction for a new sample (a VUS for instance) and obtain LIME explanations

        Parameters
        ----------
        original_X: numpy array
            Array of size n x 2623 of the original patients and controls of the suspected syndrome:
            the VGG-Face feature vector and one cell with a list of the HPO IDs
        original_y: numpy array
            The labels (usually 0 for control and 1 for patient) of the suspected syndrome
        img: str
            Path to image of new sample
        hpo_all_new_sample: list
            List of HPO IDs of new sample
        lime_iter: int
            Number of LIME iterations for the generation of facial heatmaps

        Returns
        -------
        preds: numpy array
            Predictions for the new sample
        exp_faces: list
            LIME explanations of the facial image
        local_pred_faces: list
            LIME prediction for this instance
        exp_hpos: list
            LIME explanations for the HPO terms
        local_pred_hpos: list
            LIME prediction for this instance
        """
        logger = logging.getLogger('PhenoScore')
        if self.mode == 'both' or self.mode == 'face':
            clf, hpo_terms_pt, hpo_terms_cont, scale_face, scale_hpo, vgg_face_pt, vgg_face_cont, x, clf_face, clf_hpo = \
            get_clf(original_x, original_y, self._simscorer, self.mode, self._facial_feature_extractor.face_vector_size)
        elif self.mode == 'hpo':
            clf, hpo_terms_pt, hpo_terms_cont, scale_face, scale_hpo, vgg_face_pt, vgg_face_cont, x, clf_face, clf_hpo = \
            get_clf(original_x, original_y, self._simscorer, self.mode, None)

        if self.mode != 'face':

            filtered_hpo = self._simscorer.filter_hpo_df(hpo_all_new_sample)

            if len(hpo_terms_pt) != len(hpo_terms_cont):
                print("WARNING: Number of HPO terms for patients and controls is not equal.")

            avg_pt, avg_cont = [], []

            for i in range(len(hpo_terms_pt)):
                hpo_terms_pt[i], hpo_terms_cont[i] = self._simscorer.filter_hpo_df(
                    hpo_terms_pt[i]), self._simscorer.filter_hpo_df(hpo_terms_cont[i])
                avg_pt.append(self._simscorer.calc_similarity(filtered_hpo, hpo_terms_pt[i]))
                avg_cont.append(self._simscorer.calc_similarity(filtered_hpo, hpo_terms_cont[i]))
            hpo_features = np.array([[np.mean(avg_pt), np.mean(avg_cont)]])
            hpo_features = scale_hpo.transform(hpo_features)
            hpo_features = np.nan_to_num(hpo_features)
            preds_hpo = clf_hpo.predict_proba(hpo_features)[:, 1]
        if self.mode != 'hpo':
            face_features = np.array(self._facial_feature_extractor.process_file(img)).reshape(1, -1)
            face_features = normalize(scale_face.transform(face_features))
            preds_face = clf_face.predict_proba(face_features)[:, 1]

        if self.mode == 'face':
            X_lime = np.append(x[:, :self._facial_feature_extractor.face_vector_size], face_features, axis=0)
            clf = clf_face
            preds_both, preds_hpo = None, None
        elif self.mode == 'hpo':
            X_lime = np.append(x[:, -1].reshape(-1, 1), np.zeros((1, 1)), axis=0)
            X_lime[len(X_lime) - 1, -1] = hpo_all_new_sample
            clf = clf_hpo
            preds_both, preds_face = None, None
        elif self.mode == 'both':
            X_lime = np.append(x, np.append(face_features, np.zeros((1, 1)), axis=1), axis=0)
            X_lime[len(X_lime) - 1, -1] = hpo_all_new_sample

            preds_both = clf.predict_proba(np.append(face_features, hpo_features, axis=1))[:, 1]

        if lime_iter > 0:
            exp_face, local_pred_face, exp_hpo, local_pred_hpo, csi, vsi, max_rsquared, kernel_width, sum_coeffs = explain_prediction(X_lime, len(X_lime) - 1, clf,
                                                                                    scale_face, scale_hpo, hpo_terms_pt,
                                                                                    hpo_terms_cont,
                                                                                    self._simscorer,
                                                                                    self._simscorer.name_to_id_and_reverse,
                                                                                    img, n_iter=lime_iter,
                                                                                    facial_feature_extractor=self._facial_feature_extractor,
                                                                                    lime_config=LIME_config,
                                                                                    optilime_config=OptiLIME_config,
                                                                                    kernel_width=kernel_width)
            self.vus_results = {
                "preds_both": preds_both,
                "preds_hpo": preds_hpo,
                "preds_face": preds_face,
                "exp_faces_all": exp_face,
                "exp_hpos_all": exp_hpo,
                "img": img,
                "csi": csi,
                "vsi": vsi,
                "max_rsquared": max_rsquared,
                "kernel_width": kernel_width,
                "sum_coeffs": sum_coeffs,
        }
        else:
            self.vus_results = {
                "preds_both": preds_both,
                "preds_hpo": preds_hpo,
                "preds_face": preds_face,
                "img": img
            }
        return self

    def gen_vus_figure(self, filename: str) -> None:
        """
        Generate the average heatmap and HPO explanation for an individual prediction, for a VUS for instance

        Parameters
        ----------
        filename: str
            Filename of exported figure

        Returns
        -------
        fig: matplotlib fig object
            Figure
        """
        preds_both = self.vus_results["preds_both"]
        preds_hpo = self.vus_results["preds_hpo"]
        preds_face = self.vus_results["preds_face"]
        exp_faces_all = self.vus_results["exp_faces_all"]
        exp_hpos_all = self.vus_results["exp_hpos_all"]
        img = self.vus_results["img"]
        csi = self.vus_results["csi"]
        vsi = self.vus_results["vsi"]
        max_rsquared = self.vus_results["max_rsquared"]

        if self.mode == 'both':
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 1, figsize=(6, 5))
            axs = np.array([axs])

        if self.mode in ('face', 'both'):
            fig = get_heatmap_from_multiple(exp_faces_all, fig, axs[0], self._facial_feature_extractor.get_norm_image(img),
                                            0.6, self._facial_feature_extractor.input_image_size)
            axs[0].set_title('Face: ' + str(np.round(np.mean(preds_face), 2)), fontsize=18, fontweight='bold')

        if self.mode in ('hpo', 'both'):
            if isinstance(exp_hpos_all, list):
                df_summ_hpo = get_top_hpo(exp_hpos_all, False)
            else:
                df_summ_hpo = get_top_hpo([exp_hpos_all], False)

            axs[-1].set_title('HPO: ' + str(np.round(np.mean(preds_hpo), 2)), fontsize=18, fontweight='bold')

            df_summ_hpo = df_summ_hpo.sort_values('corr', ascending=False)

            if len(df_summ_hpo) > 0:
                g = sns.barplot(x=df_summ_hpo['corr'], y=list(df_summ_hpo.index), color='blue', alpha=0.6, ax=axs[-1])

                for bar in g.patches:
                    if bar.get_width() < 0:
                        bar.set_color('red')
                df_summ_hpo = df_summ_hpo.reset_index(drop=True)
                for y in range(len(df_summ_hpo)):
                    axs[-1].text(0, y, df_summ_hpo.loc[y, 'hpo'] + ' = ' + str(int(df_summ_hpo.loc[y, 'positive'])),
                                fontsize=12, horizontalalignment='center', verticalalignment='center',
                                fontweight='semibold')
                axs[-1].set_xlabel('LIME regression coefficient')
                if csi > 0:
                    axs[-1].text(-0.25, -2, f'CSI = {csi}%', fontsize=12, horizontalalignment='left')
                    axs[-1].text(0, -2, f'VSI = {vsi}%', fontsize=12, horizontalalignment='center')
                if max_rsquared > 0:
                    axs[-1].text(0.25, -2, rf'$R^2={round(max_rsquared, 1)}$', fontsize=12, horizontalalignment='right')
                axs[-1].set_xlim(-0.25, 0.25)
                axs[-1].set_yticks([])
            else:
                axs[-1].text(0.25, 0.5, 'No relevant features found')
            axs[-1].axes.yaxis.set_visible(False)
            axs[-1].spines['left'].set_visible(False)
            axs[-1].spines['right'].set_visible(False)
            axs[-1].spines['top'].set_visible(False)
        if self.mode == 'both':
            fig.suptitle('PhenoScore: ' + str(np.round(np.mean(preds_both), 2)), fontsize=20, fontweight='bold')
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print("Figure saved as " + filename)
        plt.show()
        return