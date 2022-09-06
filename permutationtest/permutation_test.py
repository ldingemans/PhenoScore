import pandas as pd 
import numpy as np
from tqdm import tqdm
import sys
from sklearn.metrics import brier_score_loss, roc_auc_score
import traceback
from models.svm import svm_class
from hpo_phenotype.calc_hpo_sim import init_calc_similarity, calc_sim_scores, calc_full_sim_mat, calc_similarity, get_base_graph, filter_hpo_df
import os

sys.setrecursionlimit(1500)

ROOT_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))

def get_loss(X, y, hpo_network, name_to_id, scorer, mode, sim_mat):
    """
    Get the predictions for current (possibly randomized) y and X
    
    Parameters
    ----------
    X: numpy array
        Array of size n x 2623 of the original patients and controls of the suspected syndrome: the VGG-Face feature vector and one cell with a list of the HPO IDs
    y: numpy array
        The y labels
    hpo_network: networkx graph
        The HPO graph as initiliazed by phenopy
    name_to_id: dict
        Dictionary that can be used to convert HPO names to HPO IDs
    scorer: phenopy scorer instance
        Scorer object that can be used to calculate semantic similarity between lists of HPO terms
    mode: str
        Whether to use facial data, HPO terms, or both
    sim_mat: numpy array
        The full calculated similarity matrix
    
    Returns
    -------
    y_real: numpy array
        The real y labels when in test cross-validation split
    y_pred: numpy array
        The predicted y labels during cross-validation
    """
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from sklearn.preprocessing import normalize, StandardScaler
    
    if len(X) < 20:
        skf = LeaveOneOut()
        skf.get_n_splits(X, y)
    else:
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(X, y)

    y_pred, y_real, y_ind = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        
        if mode != 'face':
            resnik_avg_train, resnik_avg_test = calc_sim_scores(sim_mat,train_index,test_index,y_train)
            if mode == 'both':
                X_face_train = np.array(X_train[:,:-1],dtype=float)
                X_face_test = np.array(X_test[:,:-1],dtype=float)
                                        
            scale = StandardScaler()
            X_hpo_train_norm = scale.fit_transform(resnik_avg_train)
            X_hpo_test_norm = scale.transform(resnik_avg_test)
        else:
            X_face_train = np.array(X_train,dtype=float)
            X_face_test = np.array(X_test,dtype=float)
            
        if mode != 'hpo':
            scale = StandardScaler()
            X_face_train_norm = normalize(scale.fit_transform(X_face_train))
            X_face_test_norm = normalize(scale.transform(X_face_test))

        if mode == 'face':
            predictions, clf = svm_class(X_face_train_norm, y_train, X_face_test_norm)
        elif mode == 'hpo':
            predictions, clf = svm_class(X_hpo_train_norm, y_train, X_hpo_test_norm)
        elif mode == 'both':
            X_train = np.append(X_face_train_norm, X_hpo_train_norm,axis=1)
            X_test = np.append(X_face_test_norm, X_hpo_test_norm,axis=1)
            predictions, clf = svm_class(X_train, y_train, X_test)

        y_pred.extend(predictions)
        y_real.extend(y_test)
        y_ind.extend(test_index)

    y_real, y_pred, y_ind = np.array(y_real), np.array(y_pred), np.array(y_ind)
    return y_real, y_pred, y_ind

def c2st(X, y, hpo_network, name_to_id, scorer, mode, bootstraps, pbar=None):
    """
    Perform Classifier Two Sample Test (C2ST) by randomizing the y labels, to obtain a p-value for the classification results.
    Inspired by Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.
    
    
    Parameters
    ----------
    X: numpy array
        Array of size n x 2623 of the original patients and controls of the suspected syndrome: the VGG-Face feature vector and one cell with a list of the HPO IDs
    y: numpy array
        The y labels
    hpo_network: networkx graph
        The HPO graph as initiliazed by phenopy
    name_to_id: dict
        Dictionary that can be used to convert HPO names to HPO IDs
    scorer: phenopy scorer instance
        Scorer object that can be used to calculate semantic similarity between lists of HPO terms
    mode: str
        Whether to use facial data, HPO terms, or both
    bootstraps: int
        Number of bootstraps to perform
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
    y_bar = np.mean(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    if mode != 'face':
        sim_mat = calc_full_sim_mat(X,y, hpo_network, name_to_id, scorer)
    else:
        sim_mat = None
        
    y_real, y_pred, y_ind = get_loss(X,y, hpo_network, name_to_id, scorer, mode, sim_mat)
    emp_loss, emp_loss_auc = brier_score_loss(y_real, y_pred), roc_auc_score(y_real, y_pred)
    pbar.update(1)
    for b in range(bootstraps):         
        y_random = generate_random_y(y_bar, len(y))
        y_real, y_pred, y_ind = get_loss(X, y_random, hpo_network, name_to_id, scorer, mode, sim_mat)
        bs_losses.append(brier_score_loss(y_real, y_pred))
        pbar.update(1)
    return emp_loss, np.array(bs_losses), emp_loss_auc

def generate_random_y(y_mean, size):
    """
    Shuffle y labels, while keeping the same ratio of positive and negative classes
    
    
    Parameters
    ----------
    y_mean: float
        Ratio of positive and negative classes to obtain
    size: int
        Length of shuffled y labels
    
    Returns
    -------
    y_rand: numpy array
        Shuffled y labels
    """
    import itertools
    if size < 10:
        y_rand = np.array(list(itertools.product([0, 1], repeat=size)))
    else:
        y_rand = np.random.binomial(1, y_mean, size=(1000, size))
    y_rand = y_rand[np.mean(y_rand,axis=1) == y_mean]
    y_rand = np.random.permutation(y_rand)[0] 
    
    assert(len(y_rand) == size)
    return y_rand    
    
def permutation_test(X, y, bootstraps_per_resample=100, mode='both'):
    """
    Do the permutation test for X and y and obtain a p-value for the classifier.
    
    Parameters
    ----------
    X: list
        List of numpy arrays. Each entry in the list is an array of size n x 2623: the VGG-Face feature vector and one cell with a list of the HPO IDs.
        The multiple entries in the list correspond to multiple matched sampled controls. If just one comparison is needed, for a fixed dataset (so no sampling of controls or similar), this can just be list of length one.
    y: numpy array
        The labels (usually 0 for control and 1 for patient) 
    bootstraps_per_resample: int
        Number of bootstraps per analysis. Is repeated per entry in X, so if those are five, in total 500 bootstraps will be performed.
    mode: str
        Whether to use facial data, HPO terms, or both
    
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
    from scipy import stats
    from scipy.stats import mannwhitneyu
    
    assert((mode == 'both') or (mode == 'face') or (mode == 'hpo'))
    
    if np.mean(y) != 0.5:
        print("WARNING: the dataset is imbalanced. This permutation test has not been validated for imbalanced datasets, it is therefore recommended to undersample the majority class. The test will however continue now.")
    
    hpo_network, name_to_id, scorer = init_calc_similarity(ROOT_DIR)   
    bootstrapped_results = []
    classifier_results = []   
    classifier_aucs = []
    ps = []
    
    pbar = tqdm(total=len(X)*bootstraps_per_resample+len(X))
        
    for z in range(len(X)):   
        acc, random_losses, auc = c2st(X[z],y,hpo_network, name_to_id, scorer,mode, bootstraps_per_resample, pbar)
        
        classifier_results.append(acc)
        classifier_aucs.append(auc)
        bootstrapped_results.extend(random_losses)
        ps.append(mannwhitneyu(acc, random_losses, alternative='less',nan_policy='raise')[1])
        
    p_value = stats.combine_pvalues(ps, method='fisher', weights=None)[1]
    
    assert len(bootstrapped_results) == (bootstraps_per_resample * len(X))
    assert len(classifier_results) == len(X)
    
    return classifier_results, bootstrapped_results, ps, p_value, classifier_aucs

def negative_control_permutation_test():
    """
    Generate random data to see if our permutation tests provides a high p value then, as a negative control.

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
    from random import shuffle
    
    hpo_network, name_to_id, scorer = init_calc_similarity(ROOT_DIR)  
    nodes = list(hpo_network.nodes())
    
    N_PATIENTS = 20
    
    X = []
    # y = np.random.binomial(1, 0.5, N_PATIENTS) #leads to imbalanced dataset, can be used for testing
    y = np.array([0] * int(N_PATIENTS / 2) + [1] * int(N_PATIENTS / 2))
    shuffle(y)
    
    for p in range(5): #simulating resampling of controls
        features_rand = np.zeros((N_PATIENTS, 2623),dtype=object)
        for i in range(len(features_rand)):
            face_rand = np.random.uniform(-1, 1, 2622)
            hpo_rand = list(np.random.choice(nodes,size=np.random.randint(3,30),replace=False))
            
            features_rand[i, :2622] = face_rand
            features_rand[i, 2622] = hpo_rand
        X.append(features_rand)
        
    classifier_results, bootstrapped_results, ps, p_value, classifier_aucs = permutation_test(X, y, bootstraps_per_resample=100)
    return classifier_results, bootstrapped_results, ps, p_value, classifier_aucs
