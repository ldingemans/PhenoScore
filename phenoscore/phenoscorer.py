import pandas as pd
import numpy as np
import ast
import os
from deepface import DeepFace
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from phenoscore.hpo_phenotype.calc_hpo_sim import SimScorer
from phenoscore.permutationtest.permutation_test import PermutationTester
from phenoscore.permutationtest.cross_validation import CrossValidatorAndLIME
from phenoscore.tables_and_figures.gen_tables_and_figs import get_top_HPO, get_heatmap_from_multiple
from phenoscore.explainability_lime.LIME import get_norm_image, explain_prediction
from phenoscore.models.svm import get_clf
from sklearn.preprocessing import normalize


class PhenoScorer:
    def __init__(self, gene_name, mode, method_hpo_similarity='Resnik', method_summ_hpo_similarities='BMA'):
        """
        Constructor

        Parameters
        ----------
        gene_name: str
            Name of gene for the LIME plot
        mode: str
            Whether to use facial data, HPO terms, or both
        method_hpo_similarity
            Scoring method to use to calculate semantic similarity.
            Can be HRSS, Resnik, Jaccard or word2vec
        method_summ_hpo_similarities: str
            Method to summarize the HPO term similarities.
            Can be BMA, BMWA, or maximum.
        """
        devices = tf.config.list_physical_devices('GPU')
        if len(devices) == 0:
            print('Using CPU, since no GPUs are found!')
        else:
            print('Using GPUs:' + str(devices))

        assert ((mode == 'both') or (mode == 'face') or (mode == 'hpo'))
        self.gene_name = gene_name
        self.mode = mode
        self.permutation_test_brier = None
        self.permutation_test_auc = None
        self.permutation_test_p_value = None
        self.lime_results = None
        self._simscorer = SimScorer(scoring_method=method_hpo_similarity, sum_method=method_summ_hpo_similarities)
        self.vus_results = None

    def load_data_from_excel(self, path_to_excel_file):
        """
        Load an excel file with the same columns as the provided examples, such as the random_generated_sample_data.xlsx

        Parameters
        ----------
        path_to_excel_file: str
            Path to excel file to process
        """
        df_data = pd.read_excel(path_to_excel_file)
        df_data['graph'], df_data['hpo_name_inc_parents'] = '', ''

        X = np.zeros((len(df_data), 2623), dtype=object)
        y = df_data.loc[:, 'y_label'].to_numpy()
        img_paths = []

        # convert images to VGG-Face feature vector and create the appropriate numpy array X
        for i in range(len(df_data)):
            if self.mode != 'hpo':
                full_img_path = os.path.join(os.path.dirname(path_to_excel_file),
                                             df_data.loc[i, 'path_to_file'].replace('\\', os.sep))
                X[i, :2622] = DeepFace.represent(full_img_path, model_name='VGG-Face', detector_backend='mtcnn')
                img_paths.append(full_img_path)
            if self.mode != 'face':
                X[i, 2622] = ast.literal_eval(df_data.loc[i, 'hpo_all'])
                df_data.at[i, 'graph'] = self._simscorer.get_graph(ast.literal_eval(df_data.loc[i, 'hpo_all']), False)
                df_data.at[i, 'hpo_name_inc_parents'] = list(df_data.at[i, 'graph'].nodes())

        return X, y, img_paths, df_data

    def permutation_test(self, X, y, bootstraps=1000):
        """

        Parameters
        ----------
        X: numpy array
            Array of size n x 2623 of the original patients and controls of the suspected
            syndrome: the VGG-Face2 feature vector and one cell with a list of the HPO IDs.
        y: numpy array
            The y labels of the data.
        bootstraps: int
            Number of bootstraps for permutation test.
        """
        permutation_tester = PermutationTester(self._simscorer, mode=self.mode, bootstraps=bootstraps)
        if type(X) == list:
            permutation_tester.permutation_test_multiple_X(X, y)
        else:
            permutation_tester.permutation_test(X, y)
        self.permutation_test_brier = np.mean(permutation_tester.classifier_results)
        self.permutation_test_auc = np.mean(permutation_tester.classifier_aucs)
        self.permutation_test_p_value = np.mean(permutation_tester.p_value)
        return self

    def get_lime(self, X, y, img_paths, n_lime=5):
        """

        Parameters
        ----------
        X: numpy array
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
        cross_val_and_limer.get_results(X, y, img_paths, self.mode, self._simscorer)
        self.lime_results = cross_val_and_limer.results
        return self

    def gen_lime_and_results_figure(self, bg_image, df_data, filename, bubble_plot=False, lime_iterations=1):
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
        mean_face_norm = get_norm_image(bg_image)[0]

        score_aroc_both = np.round(self.lime_results.loc[0, 'roc_both_svm'], 2)
        score_aroc_face = np.round(self.lime_results.loc[0, 'roc_face_svm'], 2)
        score_aroc_hpo = np.round(self.lime_results.loc[0, 'roc_hpo_svm'], 2)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs = axs.flatten()

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

        if self.mode != 'hpo':
            df_top = df_exp[df_exp['face_exp'] != ''].reset_index(drop=True)
        else:
            df_top = df_exp[df_exp['hpo_exp'] != ''].reset_index(drop=True)

        if self.mode != 'face':
            df_summ_hpo = get_top_HPO(df_top.loc[:, 'hpo_exp'].explode(), True)

        if self.mode != 'hpo':
            fig = get_heatmap_from_multiple(df_top.loc[:, 'face_exp'].explode(), fig, axs[0], mean_face_norm, 0.5)
            axs[0].set_title('Face: ' + str(score_aroc_face), fontsize=18, fontweight='bold')
            axs[0].set_ylabel(self.gene_name, fontsize=18, fontweight='bold', style='italic')
            axs[0].annotate("n=" + str(int(np.sum(self.lime_results.loc[0, 'y_real'] == 1)/lime_iterations)), (95, 240),
                            annotation_clip=False,
                            fontsize=16, fontweight='bold')

        if self.mode != 'face':
            axs[1].set_title('HPO: ' + str(score_aroc_hpo), fontsize=18, fontweight='bold')

            df_summ_hpo = df_summ_hpo.sort_values('corr', ascending=False)
            if bubble_plot == True:
                df_summ_hpo = df_summ_hpo.loc[df_summ_hpo['corr'] > 0, :] # only use positive correlations for this plot
                x = [0] * len(df_summ_hpo)
                y = np.flip(range(len(df_summ_hpo)))
                axs[1].scatter(x, y, alpha=0.5, s=df_summ_hpo['corr'].to_numpy() * 30000)
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
                    axs[1].annotate(txt, (x[i] + 0.12, y[i]), verticalalignment="center",
                                           horizontalalignment="left", fontsize=16, fontweight='semibold')
                axs[1].set_xlim(-0.2, 0.5)
                axs[1].set_ylim(-2, 6)
                axs[1].set_axis_off()
            else:
                for hpo_term in df_summ_hpo['hpo']:
                    df_summ_hpo.loc[hpo_term, 'prev_0'] = df_data.loc[
                        df_data['y_label'] == 0, 'hpo_name_inc_parents'].astype(str).str.contains(hpo_term).mean()
                    df_summ_hpo.loc[hpo_term, 'prev_1'] = df_data.loc[
                        df_data['y_label'] == 1, 'hpo_name_inc_parents'].astype(str).str.contains(hpo_term).mean()

                g = sns.barplot(x=df_summ_hpo['corr'], y=list(df_summ_hpo.index), color='blue', alpha=0.6, ax=axs[1])

                for bar in g.patches:
                    if bar.get_width() < 0:
                        bar.set_color('red')
                axs[1].set_xlim(-0.25, 0.25)
                axs[1].set_yticks([])
                axs[1].axes.yaxis.set_visible(False)
                axs[1].spines['left'].set_visible(False)
                axs[1].spines['right'].set_visible(False)
                axs[1].spines['top'].set_visible(False)
                axs[1].set_xlabel('LIME regression coefficient')
                df_summ_hpo = df_summ_hpo.reset_index(drop=True)
                for y in range(len(df_summ_hpo)):
                    axs[1].text(0, y, df_summ_hpo.loc[y, 'hpo'], fontsize=12, horizontalalignment='center',
                                verticalalignment='center', fontweight='semibold')
                    axs[1].text(-0.28, y, str(int(np.round(df_summ_hpo.loc[y, 'prev_0'] * 100))) + '%', fontsize=12,
                                horizontalalignment='left', verticalalignment='center')
                    axs[1].text(0.28, y, str(int(np.round(df_summ_hpo.loc[y, 'prev_1'] * 100))) + '%', fontsize=12,
                                horizontalalignment='right', verticalalignment='center')

        fig.suptitle('PhenoScore: ' + str(score_aroc_both), fontsize=20, fontweight='bold', horizontalalignment='center', x=0.53)
        plt.subplots_adjust(top=0.77)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print("Figure saved as " + filename)
        plt.show()
        return

    def predict_new_sample(self, original_X, original_y, img, hpo_all_new_sample, lime_iter=100):
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
        clf, hpo_terms_pt, hpo_terms_cont, scale_face, scale_hpo, vgg_face_pt, vgg_face_cont, X, clf_face, clf_hpo = \
            get_clf(original_X, original_y, self._simscorer, self.mode)

        if self.mode != 'face':
            filtered_hpo = self._simscorer.filter_hpo_df(hpo_all_new_sample)

            assert len(hpo_terms_pt) == len(hpo_terms_cont)

            avg_pt, avg_cont = [], []

            for i in range(len(hpo_terms_pt)):
                avg_pt.append(self._simscorer.calc_similarity(filtered_hpo, hpo_terms_pt[i]))
                avg_cont.append(self._simscorer.calc_similarity(filtered_hpo, hpo_terms_cont[i]))

            hpo_features = np.array([[np.mean(avg_pt), np.mean(avg_cont)]])
            hpo_features = scale_hpo.transform(hpo_features)
            preds_hpo = clf_hpo.predict_proba(hpo_features)[:, 1]

        if self.mode != 'hpo':
            face_features = np.array(DeepFace.represent(img, model_name='VGG-Face', detector_backend='mtcnn')).reshape(1, -1)
            face_features = normalize(scale_face.transform(face_features))
            preds_face = clf_face.predict_proba(face_features)[:, 1]

        if self.mode == 'face':
            X_lime = np.append(X[:, :2622], face_features, axis=0)
            clf = clf_face
            preds_both, preds_hpo = None, None
        elif self.mode == 'hpo':
            X_lime = np.append(X[:, -1].reshape(-1, 1), np.zeros((1, 1)), axis=0)
            X_lime[len(X_lime) - 1, -1] = hpo_all_new_sample
            clf = clf_hpo
            preds_both, preds_face = None, None
        elif self.mode == 'both':
            X_lime = np.append(X, np.append(face_features, np.zeros((1, 1)), axis=1), axis=0)
            X_lime[len(X_lime) - 1, -1] = hpo_all_new_sample

            preds_both = clf.predict_proba(np.append(face_features, hpo_features, axis=1))[:, 1]

        exp_face, local_pred_face, exp_hpo, local_pred_hpo = explain_prediction(X_lime, len(X_lime) - 1, clf,
                                                                                scale_face, scale_hpo, hpo_terms_pt,
                                                                                hpo_terms_cont,
                                                                                self._simscorer,
                                                                                self._simscorer.name_to_id_and_reverse,
                                                                                img, n_iter=lime_iter)
        self.vus_results = [preds_both, preds_hpo, preds_face, exp_face, exp_hpo, img]
        return self

    def gen_vus_figure(self, filename):
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
        [preds_both, preds_hpo, preds_face, exp_faces_all, exp_hpos_all, img] = self.vus_results

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs = axs.flatten()

        fig = get_heatmap_from_multiple(exp_faces_all, fig, axs[0], get_norm_image(img)[0], 0.6)
        axs[0].set_title('Face: ' + str(np.round(preds_face, 2)), fontsize=18, fontweight='bold')

        df_summ_hpo = get_top_HPO([exp_hpos_all], False)

        axs[1].set_title('HPO: ' + str(np.round(preds_hpo, 2)), fontsize=18, fontweight='bold')

        df_summ_hpo = df_summ_hpo.sort_values('corr', ascending=False)

        if len(df_summ_hpo) > 0:
            g = sns.barplot(x=df_summ_hpo['corr'], y=list(df_summ_hpo.index), color='blue', alpha=0.6, ax=axs[1])

            for bar in g.patches:
                if bar.get_width() < 0:
                    bar.set_color('red')
            df_summ_hpo = df_summ_hpo.reset_index(drop=True)
            for y in range(len(df_summ_hpo)):
                axs[1].text(0, y, df_summ_hpo.loc[y, 'hpo'] + ' = ' + str(int(df_summ_hpo.loc[y, 'positive'])),
                            fontsize=12, horizontalalignment='center', verticalalignment='center',
                            fontweight='semibold')
            axs[1].set_xlabel('LIME regression coefficient')

            axs[1].set_xlim(-0.25, 0.25)
            axs[1].set_yticks([])
        else:
            axs[1].text(0.25, 0.5, 'No relevant features found')
        axs[1].axes.yaxis.set_visible(False)
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        fig.suptitle('PhenoScore: ' + str(np.round(preds_both, 2)), fontsize=20, fontweight='bold')
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print("Figure saved as " + filename)
        plt.show()
        return
