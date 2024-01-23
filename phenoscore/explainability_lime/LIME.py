import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from lime import lime_image
from sklearn.preprocessing import MultiLabelBinarizer
import traceback
from lime.lime_tabular import LimeTabularExplainer
#LIME Module with all corresponding functions


def random_mask(image):
    """
    Generate a random mask for a image

    Parameters
    ----------
    image: numpy array
        Image for which the random mask needs to be generated

    Returns
    -------
    Generated random mask
    """
    img_length = image.shape[0]
    chunk_length = int(img_length/9)

    total_pixels = int(img_length * img_length)
    n_masks = int(np.ceil(total_pixels / chunk_length))
    shifted_mask = np.zeros(np.random.randint(chunk_length), dtype=int)
    shifted_mask = np.append(shifted_mask, np.repeat(list(range(1, n_masks + 1)), chunk_length))
    shifted_mask = shifted_mask[:total_pixels].reshape(img_length, img_length)
    shifted_mask = np.repeat(shifted_mask, chunk_length, axis=0)
    shifted_mask = shifted_mask[np.random.randint(chunk_length):, :]
    shifted_mask = shifted_mask[:img_length, :img_length]

    return np.array(shifted_mask, dtype=int)


def predict_image(X_face, classifier_args):
    """
    Get predictions for a given image, or filepath. Can be used by LIME to get predictions for perturbated images

    Parameters
    ----------
    X_face: numpy array/list
        Can be a numpy array of normalized VGG-Face feature vectors, a list of file paths or a single file path
    classifier_args: dict
        Dictionary with arguments to pass to classifier while using LIME

    Returns
    -------
    predictions: numpy array
        Predictions per class
    """
    failed_images = []

    if type(X_face) == str:
        X_face = classifier_args['face_model'].get_norm_image(X_face)
    elif type(X_face) == list:
        X_face_temp = []
        for i, file in enumerate(X_face):
            try:
                X_face_temp.append(classifier_args['face_model'].get_norm_image(file))
            except ValueError as e:
                if 'Face could not be detected.' in str(e):
                    X_face_temp.append(np.zeros((1,224,224,3)))
                    failed_images.append(i)
                else:
                    raise(e)
        X_face = np.array(X_face_temp)

    img_test = classifier_args['face_model'].predict_aligned_img(X_face)

    face_features = normalize(classifier_args['scale_face'].transform(img_test))

    if 'X_hpo' in classifier_args:
        hpo_features = calc_hpo_test(classifier_args['X_hpo'], classifier_args)
        hpo_features = np.repeat(hpo_features,len(face_features),axis=0) #since we are iterating the facial features in this LIME instance, duplicate the HPO similarity to be the same in all instances

        hpo_features = classifier_args['scale_hpo'].transform(hpo_features)

        predictions = classifier_args['clf'].predict_proba(np.append(face_features,hpo_features,axis=1))
        predicted_classes = classifier_args['clf'].predict(np.append(face_features,hpo_features,axis=1))
    else:
        predictions = classifier_args['clf'].predict_proba(face_features)
        predicted_classes = classifier_args['clf'].predict(face_features)

    if len(failed_images) > 0:
        predicted_classes[np.array(failed_images)] = -1
        predictions[np.array(failed_images),:] = np.nan
        print("There were images in which a face was not detected and therefore the image was not processed. Predictions are np.nan for that instance, please check.")
    return predictions


def calc_hpo_test(filtered_hpo, classifier_args):
    """
    Calculate the semantic similarity for a new (test) set of HPO terms

    Parameters
    ----------
    filtered_hpo: list
        Filtered HPO terms
    classifier_args: dict
        Dictionary with arguments to pass to classifier while using LIME

    Returns
    -------
    hpo_features: numpy array
        Calculated semantic similarities: average when compared to patients and average when compared to controls
    """
    filtered_hpo = pd.DataFrame(filtered_hpo)
    if filtered_hpo.shape[1] == 1:
        filtered_hpo = filtered_hpo.T
    filtered_hpo.columns = classifier_args['mlb_classes']

    these_hpos = []
    for i in range(len(filtered_hpo)):
        these_hpos.append(list(filtered_hpo.loc[0,filtered_hpo.loc[i,:] == 1].index))

    hpo_features = []
    for i in range(len(these_hpos)):
        avg_pt, avg_cont = [], []

        for y in range(len(classifier_args['hpo_terms_pt'])):
            avg_pt.append(classifier_args['simscorer'].calc_similarity(these_hpos[i], classifier_args['hpo_terms_pt'][y]))
        for y in range(len(classifier_args['hpo_terms_cont'])):
            avg_cont.append(classifier_args['simscorer'].calc_similarity(these_hpos[i], classifier_args['hpo_terms_cont'][y]))
        hpo_features.append([np.mean(avg_pt), np.mean(avg_cont)])

    hpo_features = np.array(hpo_features)
    return hpo_features


def predict_hpo(X_hpo, classifier_args):
    """
    Get predictions for a set of HPO terms. Can be used by LIME to get predictions for perturbated HPO terms

    Parameters
    ----------
    X_hpo: numpy array
        HPO terms to pertube
    classifier_args: dict
        Dictionary with arguments to pass to classifier while using LIME

    Returns
    -------
    preds: numpy array
        Predictions per class
    """
    hpo_features = calc_hpo_test(X_hpo, classifier_args)
    hpo_features = classifier_args['scale_hpo'].transform(hpo_features)

    img = classifier_args['X_face']

    if img is not None:
        if type(img) == str:
            face_features = np.array(classifier_args['face_model'].process_file(img)).reshape(1,-1)
        else:
            face_features = img

        if face_features.ndim == 1:
            face_features = face_features.reshape(1, -1)

        face_features = np.repeat(face_features,len(hpo_features),axis=0) #since we are iterating the HPO features
        # in this LIME instance, duplicate the facial features to be the same in all instances
        face_features = normalize(classifier_args['scale_face'].transform(face_features))
        preds = classifier_args['clf'].predict_proba(np.append(face_features, hpo_features,axis=1))
    else:
        preds = classifier_args['clf'].predict_proba(hpo_features)
    return preds


def explain_prediction(X, index_pt, clf, scale_face=None, scale_hpo=None, hpo_terms_pt=None, hpo_terms_cont=None,
                       simscorer=None, id_to_name=None, img_path_index_patient=None, n_iter=100, facial_feature_extractor=None):
    """
    Use LIME to generate predictions for a prediction. Use both image (when available) and HPO terms. When no image is available (img_path_index_patient = None), only generate LIME for HPO terms.

    Parameters
    ----------
    X_processed:
        The original input data, without the converted X - in the converted X, the HPO IDs are replaced with the average semantic similarity with patients and controls

    clf: sklearn instance
        The trained support vector machine
    scale_face: sklearn StandardScaler
        The scaler instance for scaling VGG-Face feature vector, so the test data can be transformed using a fitted scaler
    scale_hpo: sklearn StandardScaler
        Same, but for scaling the HPO features (after averaging them for patients and controls, so this is on a nx2 array)
    hpo_terms_pt: numpy array
        The HPO IDs of the patients of the investigated syndrome. These are needed seperately, because if we want to make a prediction for
        a new sample, we need to be able to calculate the semantic similarity using the original HPO IDs, before they are converted to an average for patients/controls.
    hpo_terms_cont: numpy array
        The HPO IDs of the controls
    vgg_face_pt: numpy array
        The original VGG-Face feature vector for the patients
    vgg_face_cont: numpy array
        The original VGG-Face feature vector for the controls
    simscorer: object of the SimScorer class
        Initialized objected of the SimScorer class of this package
    id_to_name: dict
        Dictionary that can be used to convert HPO IDs to HPO names
    img_path_index_patient: str
        Path to image of the patient of interest. If None, do not generate LIME explanations for the facial image, only for the HPO terms.
    n_iter: int
        Number of iterations to use while generating LIME explanations for the image
    facial_feature_extractor: FacialFeatureExtractor object
        Instance to extract facial features, default is VGGFace, can be QMagFace as well

    Returns
    -------
    preds: numpy array
        Predictions for the new sample
    exp_face: list
        LIME explanations of the facial image
    local_pred_face: float
        LIME prediction for this instance
    exp_hpo: LIME explanation
        LIME explanations for the HPO terms
    local_pred_hpos: float
        LIME prediction for this instance
    """

    if hpo_terms_pt is not None:
        if hpo_terms_pt.ndim == 1:
            hpo_terms_pt = hpo_terms_pt.reshape(-1, 1)
        if hpo_terms_cont.ndim == 1:
            hpo_terms_cont = hpo_terms_cont.reshape(-1, 1)
        #the problem we have is that LIME only works with tabular data, while HPO is a graph.
        # So we first expand the HPO data into tabular, and later compress it again into a list
        # that can be used for the Resnik score
        X_hpo_pure_train = np.append(hpo_terms_pt, hpo_terms_cont, axis=0)
        mlb = MultiLabelBinarizer()
        mlb.fit_transform(X[:,-1])
        X_expanded_train = pd.DataFrame(mlb.transform(X_hpo_pure_train[:,0]),columns=mlb.classes_)
        y_train = np.array([1] * len(hpo_terms_pt) + [0] * len(hpo_terms_cont))
        X_expanded_test = []
        for class_ in mlb.classes_:
            X_expanded_test.append(int(class_ in X[index_pt,-1]))
        classifier_args = {'X_face'         : np.array(X[index_pt, :-1], dtype=float),
                           'X_hpo'          : np.array(X_expanded_test),
                           'clf'            : clf,
                           'scale_face'     : scale_face,
                           'scale_hpo'      : scale_hpo,
                           'hpo_terms_pt'   : hpo_terms_pt[:,0],
                           'hpo_terms_cont' : hpo_terms_cont[:,0],
                           'mlb_classes'    : mlb.classes_,
                           'simscorer'      : simscorer,
                           'face_model'     : facial_feature_extractor}
    else:
        classifier_args = {'X_face'         : np.array(X[index_pt, :-1], dtype=float),
                           'clf'            : clf,
                           'scale_face'     : scale_face,
                           'face_model'     : facial_feature_extractor}

    if img_path_index_patient is not None:
        explainer = lime_image.LimeImageExplainer(verbose=False, feature_selection='lasso_path')
        exp_face = []
        local_pred_face = []
        segmentation_fn = random_mask
        aligned_image = facial_feature_extractor.get_norm_image(img_path_index_patient)
        for m in range(n_iter):
            try:
                explanation = explainer.explain_instance(aligned_image, predict_image, num_samples=200, batch_size=50,
                                                         segmentation_fn=segmentation_fn, classifier_args=classifier_args)
                exp_face.append(explanation)
                local_pred_face.append(explanation.local_pred[0])
            except:
                print(traceback.format_exc())
    else:
        classifier_args['X_face'] = None
        exp_face, local_pred_face = np.nan, np.nan

    if hpo_terms_pt is not None:
        feature_hpo_names = []
        for hpo_id in mlb.classes_:
            feature_hpo_names.append(id_to_name[hpo_id.strip()])

        explainer = LimeTabularExplainer(X_expanded_train.to_numpy(),
            training_labels=y_train,
            categorical_features=np.array(range(len(mlb.classes_))),
            feature_names = feature_hpo_names,
            feature_selection='lasso_path',
            verbose=False,
            mode='classification')
        # Now explain a prediction
        for attempt in range(10):
            try:
                exp_hpo = explainer.explain_instance(np.array(X_expanded_test), predict_hpo, num_samples=1000,
                                                     classifier_args=classifier_args)
            except:
                print(traceback.format_exc())
                continue
            else:
                break
        local_pred_hpo = exp_hpo.local_pred[0]
    else:
        exp_hpo, local_pred_hpo = np.nan, np.nan

    return exp_face, local_pred_face, exp_hpo, local_pred_hpo
