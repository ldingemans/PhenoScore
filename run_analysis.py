import os
from phenoscore.phenoscore import PhenoScorer

if __name__ == '__main__':
        phenoscorer = PhenoScorer(gene_name='SATB1', mode='both', method_hpo_similarity='Resnik', method_summ_hpo_similarities='BMA')
        X, y, img_paths, df_data = phenoscorer.load_data_from_excel(os.path.join("phenoscore", "sample_data",
                                                                                 "satb1_data.xlsx"))
        print('Data loaded!')
        phenoscorer.permutation_test(X, y, bootstraps=1000)
        print("Brier:" + str(phenoscorer.permutation_test_brier))
        print("AUC:" + str(phenoscorer.permutation_test_auc))
        print("P value:" + str(phenoscorer.permutation_test_p_value))

        phenoscorer.get_lime(X, y, img_paths, n_lime=5)
        phenoscorer.gen_lime_and_results_figure(bg_image=os.path.join("phenoscore", "sample_data", "background_image.jpg"),
                                                df_data=df_data, filename='lime_figure.pdf')
        print("LIME images generated!")

        #lets pretend a SATB1 indvidual is a VUS and we want a prediction for SATB1
        phenoscorer.predict_new_sample(X, y, img_paths[-1], X[-1,-1])
        print("Predictive score between 0 (control) and 1 (syndrome): " + str(phenoscorer.vus_results[0]))
        phenoscorer.gen_vus_figure(filename='individual_lime_explanations.pdf')
