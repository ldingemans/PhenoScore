import traceback
from lime import lime_image
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
import os
import sys
import ast
import tensorflow as tf

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import normalize, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut, train_test_split

from models.svm import svm_class
from hpo_phenotype.calc_hpo_sim import init_calc_similarity, calc_sim_scores, calc_full_sim_mat, calc_similarity, get_base_graph, filter_hpo_df, get_graph
from explainability_lime.LIME import predict_hpo, predict_image, get_norm_image, random_mask, explain_prediction, draw_heatmap
from explainability_lime.VUS_prediction import get_clf, predict_new_sample
from permutationtest.permutation_test import permutation_test, negative_control_permutation_test, get_loss
from tables_and_figures.gen_tables_and_figs import plot_incremental, gen_lime_and_results_figure, gen_vus_figure

ROOT_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))

def get_results_incremental(X, y):
    """
    Get the incremental results of the classifier by starting with a training set of size 2 and incrementally adding one patient and control
    
    Parameters
    ----------
    X: numpy array
        Array of size n x 2623: the VGG-Face feature vector and one cell with a list of the HPO IDs
    y: numpy array
        The labels (usually 0 for control and 1 for patient)    
    
    Returns
    -------
    df_results_per_p: pandas DataFrame
        DataFrame with the results
    """
    from sklearn import svm
    
    param_grid = {'C': [1e-5, 1e-3, 1, 1e3, 1e5]}

    df_results_per_p = pd.DataFrame()
    
    sim_mat = calc_full_sim_mat(X,y, hpo_network, name_to_id, scorer)
    
    p_max = len(X)
    if p_max > 50:
        p_max = 50
    
    indices = np.arange(len(X))
    
    for p in range(4,p_max,2):
        X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, indices, train_size=p, random_state=42,stratify=y)
        y_pred_all = np.zeros((len(X_test),3))
        
        y_real_test = []
        
        y_real_test.extend(y_test)
        
        sim_mat_train, sim_mat_test = calc_sim_scores(sim_mat,train_index,test_index,y_train)

        scale = StandardScaler()
        X_face_train_norm = normalize(scale.fit_transform(np.array(X_train[:,:-1],dtype=float)))
        X_face_test_norm = normalize(scale.transform(np.array(X_test[:,:-1],dtype=float)))

        scale = StandardScaler()
        X_hpo_train_norm = scale.fit_transform(sim_mat_train)
        X_hpo_test_norm = scale.transform(sim_mat_test)

        X_train = np.append(X_face_train_norm, X_hpo_train_norm,axis=1)
        X_test = np.append(X_face_test_norm, X_hpo_test_norm,axis=1)
        
        clf = GridSearchCV(
            svm.SVC(probability=True), param_grid, cv=LeaveOneOut(), scoring='neg_brier_score'
            )
        
        clf.fit(X_train, y_train)
        y_pred_all[:,2] = predictions = clf.predict_proba(X_test)[:,1]

        results = np.ones(7,dtype=object)

        results[0], results[1] = roc_auc_score(y_real_test,y_pred_all[:,0]),brier_score_loss(y_real_test,y_pred_all[:,0])
        results[2], results[3] = roc_auc_score(y_real_test,y_pred_all[:,1]),brier_score_loss(y_real_test,y_pred_all[:,1])
        results[4], results[5] = roc_auc_score(y_real_test,y_pred_all[:,2]),brier_score_loss(y_real_test,y_pred_all[:,2])
        results[6] = len(X_train)  /2

        df_results_per_p  = pd.concat([df_results_per_p , pd.DataFrame(results).T])
    df_results_per_p.columns = ['roc_face', 'brier_face', 'roc_hpo', 'brier_hpo', 'roc_both', 'brier_both', 'n_patients']
    return df_results_per_p 
    
def get_results_loo_cv(X, y, file_paths, mode, hpo_network, name_to_id, scorer, graph, id_to_name, n_lime=5):
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
    hpo_network: networkx graph
        The HPO graph as initiliazed by phenopy
    name_to_id: dict
        Dictionary that can be used to convert HPO names to HPO IDs
    scorer: phenopy scorer instance
        Scorer object that can be used to calculate semantic similarity between lists of HPO terms
    graph: networkx graph
        The HPO graph
    id_to_name: dict
        Dictionary that can be used to convert HPO IDs to HPO names
    n_lime: int
        Number of top predictions to generate LIME predictions for
    
    Returns
    -------
    df_results: pandas DataFrame
        DataFrame with the results
    """
    assert((mode == 'both') or (mode == 'face') or (mode == 'hpo'))
    
    y_pred_all_svm = np.zeros((len(X),3))
    
    if mode != 'face':
        sim_mat = calc_full_sim_mat(X,y, hpo_network, name_to_id, scorer)
    
    print("Starting cross validation procedure to compare using facial/HPO data only with PhenoScore.")
    
    if mode != 'hpo':
        y_pred_indexer = 0
        y_real_test, y_pred_all_svm[:, y_pred_indexer], y_test_ind = get_loss(X[:,:2622], y, hpo_network, name_to_id, scorer, mode='face', sim_mat=None)
    if mode != 'face':
        y_pred_indexer = 1
        y_real_test, y_pred_all_svm[:, y_pred_indexer], y_test_ind = get_loss(X, y, hpo_network, name_to_id, scorer, mode='hpo', sim_mat=sim_mat)
    if mode == 'both':
        y_pred_indexer = 2
        y_real_test, y_pred_all_svm[:, y_pred_indexer], y_test_ind = get_loss(X, y, hpo_network, name_to_id, scorer, mode='both', sim_mat=sim_mat)
    
    print("Finished cross validation and evaluation of model scores. Now starting LIME for the top " + str(n_lime) + " predictions to generate heatmaps and visualise phenotypic differences.")
    #get the LIME predictions for the top n_top, default is 5: can do for all, but takes a long time (especially facial images)

    explanations_face, explanations_hpo, local_preds_face, local_preds_hpo, file_paths_sorted = [], [], [], [], []
    
    phenoscore = pd.Series(y_pred_all_svm[:, y_pred_indexer])
    highest_predictions = phenoscore[np.array(y_real_test) == 1].nlargest(n_lime)
    highest_predictions = y_test_ind[np.array(highest_predictions.index)]

    pbar = tqdm(total=n_lime)
    
    for z in y_test_ind:
        if z in highest_predictions:
            assert(y[z] == 1)
            #we need to retrain the model to get clf and other variables with this instance in the test set and rest as training data
            test_index = np.array([z])
            all_indices = np.array(range(len(y)))
            train_index = all_indices[all_indices != test_index]
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            if mode != 'face':
                sim_mat_train, sim_mat_test = calc_sim_scores(sim_mat,train_index,test_index,y_train)  
                hpo_terms_pt, hpo_terms_cont = X_train[y_train==1, -1].reshape(-1,1), X_train[y_train==0, -1].reshape(-1,1)
                scale_hpo = StandardScaler()
                X_hpo_train_norm = scale_hpo.fit_transform(sim_mat_train)
                X_hpo_test_norm = scale_hpo.transform(sim_mat_test)
          
            if mode != 'hpo':
                X_face_train = np.array(X_train[:,:-1],dtype=float)
                X_face_test = np.array(X_test[:,:-1],dtype=float)
                
                scale_face = StandardScaler()
                X_face_train_norm = normalize(scale_face.fit_transform(X_face_train))
                X_face_test_norm = normalize(scale_face.transform(X_face_test))
            
            if mode == 'both':
                X_train = np.append(X_face_train_norm, X_hpo_train_norm,axis=1)
                X_test = np.append(X_face_test_norm, X_hpo_test_norm,axis=1)
            
            if mode == 'hpo':
                preds_svm_hpo, clf = svm_class(X_hpo_train_norm, y_train, X_hpo_test_norm)
            elif mode == 'face':
                preds_svm_face, clf = svm_class(X_face_train_norm, y_train, X_face_test_norm)
            elif mode == 'both':
                preds_svm_both, clf = svm_class(X_train,y_train, X_test)
        
            if mode == 'hpo':
                exp_face, local_pred_face, exp_hpo, local_pred_hpo = explain_prediction(X, z, clf, None, scale_hpo, hpo_terms_pt, hpo_terms_cont, hpo_network, name_to_id, scorer, id_to_name)
            elif mode == 'both':
                exp_face, local_pred_face, exp_hpo, local_pred_hpo = explain_prediction(X, z, clf, scale_face, scale_hpo, hpo_terms_pt, hpo_terms_cont, hpo_network, name_to_id, scorer, id_to_name, str(file_paths[z]), n_iter=100)
            elif mode == 'face':    
                exp_face, local_pred_face, exp_hpo, local_pred_hpo = explain_prediction(X, z, clf, scale_face, img_path_index_patient=str(file_paths[z]), n_iter=100)
                
            explanations_face.append(exp_face)
            explanations_hpo.append(exp_hpo)
            local_preds_face.append(local_pred_face)
            local_preds_hpo.append(local_pred_hpo)
            if mode != 'hpo':
                file_paths_sorted.append(file_paths[z])
            pbar.update(1)
        else:
            explanations_face.append('')
            explanations_hpo.append('')
            local_preds_face.append('')
            local_preds_hpo.append('')
            file_paths_sorted.append('')
            
    pbar.close()
    
    results = np.ones(13, dtype=object)

    results[0], results[1] = roc_auc_score(y_real_test,y_pred_all_svm[:,0]),brier_score_loss(y_real_test,y_pred_all_svm[:,0])
    results[2], results[3] = roc_auc_score(y_real_test,y_pred_all_svm[:,1]),brier_score_loss(y_real_test,y_pred_all_svm[:,1])
    results[4], results[5] = roc_auc_score(y_real_test,y_pred_all_svm[:,2]),brier_score_loss(y_real_test,y_pred_all_svm[:,2])
    results[6], results[7], results[8], results[9], results[10], results[11] = explanations_face, explanations_hpo, local_preds_face, local_preds_hpo, y_pred_all_svm[:,2], y_real_test
    results[12] = file_paths_sorted
    
    df_results = pd.DataFrame(results).T
    df_results.columns = ['roc_face_svm', 'brier_face_svm', 'roc_hpo_svm', 'brier_hpo_svm', 'roc_both_svm', 
                        'brier_both_svm', 'face_explanation', 'hpo_explanation', 'face_pred', 'hpo_pred', 'svm_preds', 'y_real', 'file_paths']            
    return df_results

if __name__ == '__main__':
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) == 0:
        print('Using CPU, since no GPUs are found!')
    else:
        print('Using GPUs:' + str(devices))
    
    hpo_network, name_to_id, scorer = init_calc_similarity(phenopy_data_directory=ROOT_DIR)    
    graph, id_to_name = get_base_graph(False)

    PHENOSCORE_MODE = 'both' #should be either face, hpo or both, depending on the data that is available
    assert((PHENOSCORE_MODE == 'both') or (PHENOSCORE_MODE == 'face') or (PHENOSCORE_MODE == 'hpo'))
    GENE_NAME = 'SATB1' #name of gene/syndrome for the plots

    #load either the randomly generated- or SATB1 data, to illustrate how the code works. Of course replace the random data with your own if needed.
    #images in the random data directory are generated using StyleGan, so are non existent persons.
    
    df_data_random = pd.read_excel(os.path.join(ROOT_DIR, "sample_data", "random_generated_sample_data.xlsx"))
    df_data = pd.read_excel(os.path.join(ROOT_DIR, "sample_data", 'satb1_data.xlsx'))
    
    df_data['graph'], df_data['hpo_name_inc_parents'] = '', ''
    
    X = np.zeros((len(df_data), 2623), dtype=object)
    y = df_data.loc[:, 'y_label'].to_numpy()
    img_paths = []
    
    #convert images to VGG-Face feature vector and create the appropriate numpy array X
    for i in range(len(df_data)):
        if PHENOSCORE_MODE != 'hpo':
            full_img_path = os.path.join(ROOT_DIR, "sample_data", df_data.loc[i,'path_to_file'].replace('\\', os.sep)) 
            X[i, :2622] = DeepFace.represent(full_img_path, model_name='VGG-Face',detector_backend='mtcnn')
            img_paths.append(full_img_path)
        if PHENOSCORE_MODE != 'face':
            X[i, 2622] = ast.literal_eval(df_data.loc[i, 'hpo_all'])
            df_data.at[i, 'graph'] = get_graph(ast.literal_eval(df_data.loc[i, 'hpo_all']), False)
            df_data.at[i, 'hpo_name_inc_parents'] = list(df_data.at[i, 'graph'].nodes())
            
    #permutation test, mode is both, only hpo or face can be specified as well
    classifier_results, bootstrapped_results, ps, p_value, classifier_aucs = permutation_test([X], y, bootstraps_per_resample = 1000, mode = PHENOSCORE_MODE)
    print("Brier:" + str(np.mean(classifier_results))); print("AUC:" + str(np.mean(classifier_aucs))); print("P value:" + str(p_value))

    #main analyses of paper
    df_results = get_results_loo_cv(X, y, np.array(img_paths), PHENOSCORE_MODE, hpo_network, name_to_id, scorer, graph, id_to_name, 5)
    gen_lime_and_results_figure(df_results, gene_name=GENE_NAME, bg_image=os.path.join(ROOT_DIR, "sample_data", "background_image.jpg"), df_data = df_data, mode = PHENOSCORE_MODE, filename = GENE_NAME + '.png')
    print("LIME images generated!")
    
    #lets pretend a random generated individual is a VUS and we want a prediction for SATB1
    random_img_path = os.path.join(ROOT_DIR, "sample_data", df_data_random.loc[1, 'path_to_file'].replace('\\', os.sep))
    pred_both, pred_hpo, pred_face, exp_faces, local_pred_faces, exp_hpos, local_pred_hpos = predict_new_sample(X, y, random_img_path, ast.literal_eval(df_data_random.loc[1, 'hpo_all']), hpo_network, name_to_id, scorer, graph, id_to_name)
    gen_vus_figure(exp_faces, [exp_hpos], random_img_path, np.mean(pred_face), np.mean(pred_hpo), np.mean(pred_both), filename='individual_lime_explanations.pdf')

    # if calculating scores per individual incrementally
    df_results_inc = get_results_incremental(X, y)
    plot_incremental(df_results_inc)
    #negative control test
    classifier_results_rand, bootstrapped_results_rand, ps_rand, p_value_rand, classifier_aucs_rand = negative_control_permutation_test()
