from hpo_phenotype.calc_hpo_sim import init_calc_similarity, get_base_graph, calc_full_sim_mat, filter_hpo_df, calc_similarity
from sklearn.preprocessing import normalize, StandardScaler
from models.svm import svm_class
from explainability_lime.LIME import explain_prediction

import numpy as np
import pandas as pd
from deepface import DeepFace


def get_clf(X, y, hpo_network, name_to_id, scorer):
    """
    Train a classifier while retaining the original scaler and features of the input data, so it can be used in LIME explanations later.
    
    Parameters
    ----------
    X: numpy array
        Array of size n x 2623: the VGG-Face feature vector and one cell with a list of the HPO IDs
    y: numpy array
        The labels (usually 0 for control and 1 for patient) 
    hpo_network: networkx graph
        The HPO graph as initiliazed by phenopy
    name_to_id: dict
        Dictionary that can be used to convert HPO names to HPO IDs
    scorer: phenopy scorer instance
        Scorer object that can be used to calculate semantic similarity between lists of HPO terms

    Returns
    -------
    clf: sklearn instance
        The trained support vector machine
    hpo_terms_pt: numpy array
        The HPO IDs of the patients of the investigated syndrome. These are needed seperately, because if we want to make a prediction for
        a new sample, we need to be able to calculate the semantic similarity using the original HPO IDs, before they are converted to an average for patients/controls.
    hpo_terms_cont: numpy array
        The HPO IDs of the controls
    scale_face: sklearn StandardScaler
        The scaler instance for scaling VGG-Face feature vector, so the test data can be transformed using a fitted scaler
    scale_hpo: sklearn StandardScaler
        Same, but for scaling the HPO features (after averaging them for patients and controls, so this is on a nx2 array)
    vgg_face_pt: numpy array
        The original VGG-Face feature vector for the patients
    vgg_face_cont: numpy array
        The original VGG-Face feature vector for the controls
    X_processed:
        The original input data, without the converted X - in the converted X, the HPO IDs are replaced with the average semantic similarity with patients and controls

    """
    X_processed = X[:,:]
    
    hpo_terms_pt = X[y==1,-1]
    hpo_terms_cont = X[y==0,-1]
    
    sim_mat = calc_full_sim_mat(X,y, hpo_network, name_to_id, scorer)
    
    sim_avg_pat = sim_mat[:,y == 1].mean(axis=1).reshape(-1,1)
    sim_avg_control = sim_mat[:,y == 0].mean(axis=1).reshape(-1,1)

    hpo_features = np.append(sim_avg_pat, sim_avg_control,axis=1)   
    face_features = np.array(X[:,:2622],dtype=float)
    
    scale_face = StandardScaler()
    face_features = normalize(scale_face.fit_transform(face_features))
    
    preds, clf_face = svm_class(face_features,y, face_features)
    
    vgg_face_pt = face_features[y==1,:]
    vgg_face_cont = face_features[y==0,:]
                        
    scale_hpo = StandardScaler()
    hpo_features = scale_hpo.fit_transform(hpo_features)
    
    preds, clf_hpo = svm_class(hpo_features,y,hpo_features)
    
    X = np.append(face_features, hpo_features,axis=1)   

    preds, clf = svm_class(X,y,X)
    return clf, hpo_terms_pt, hpo_terms_cont, scale_face, scale_hpo, vgg_face_pt, vgg_face_cont, X_processed, clf_face, clf_hpo

def predict_new_sample(original_X, original_y, img, hpo_all_new_sample, hpo_network, name_to_id, scorer, graph, id_to_name):
    """
    Train a classifier, get prediction for a new sample (a VUS for instance) and obtain LIME explanations
    
    Parameters
    ----------
    original_X: numpy array
        Array of size n x 2623 of the original patients and controls of the suspected syndrome: the VGG-Face feature vector and one cell with a list of the HPO IDs
    original_y: numpy array
        The labels (usually 0 for control and 1 for patient) of the suspected syndrome
    img: str
        Path to image of new sample
    hpo_all_new_sample: list
        List of HPO IDs of new sample
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
    clf, hpo_terms_pt, hpo_terms_cont, scale_face, scale_hpo, vgg_face_pt, vgg_face_cont, X, clf_face, clf_hpo = get_clf(original_X, original_y, hpo_network, name_to_id, scorer)
    filtered_hpo = filter_hpo_df(hpo_all_new_sample)
  
    assert len(hpo_terms_pt) == len(hpo_terms_cont)
  
    avg_pt, avg_cont = [], []
  
    for i in range(len(hpo_terms_pt)):
        avg_pt.append(calc_similarity(filtered_hpo, hpo_terms_pt[i], hpo_network, name_to_id, scorer))
        avg_cont.append(calc_similarity(filtered_hpo, hpo_terms_cont[i], hpo_network, name_to_id, scorer))
  
    hpo_features = np.array([[np.mean(avg_pt), np.mean(avg_cont)]])
    face_features = np.array(DeepFace.represent(img, model_name='VGG-Face',detector_backend='mtcnn')).reshape(1,-1)
        
    X_lime = np.append(X, np.append(face_features,np.zeros((1,1)),axis=1),axis=0)
    X_lime[len(X_lime) -1, -1] = hpo_all_new_sample
  
    face_features = normalize(scale_face.transform(face_features))
    hpo_features = scale_hpo.transform(hpo_features)
    
    preds_face = clf_face.predict_proba(face_features)[:,1]
    preds_hpo = clf_hpo.predict_proba(hpo_features)[:,1]
    preds_both = clf.predict_proba(np.append(face_features, hpo_features,axis=1))[:,1]
    
    exp_face, local_pred_face, exp_hpo, local_pred_hpo = explain_prediction(X_lime, len(X_lime)-1, clf, scale_face, scale_hpo, hpo_terms_pt, hpo_terms_cont, hpo_network, name_to_id, scorer, id_to_name, img,n_iter=100)

    return preds_both, preds_hpo, preds_face, exp_face, local_pred_face, exp_hpo, local_pred_hpo
