import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import normalize, StandardScaler


def get_clf(X, y, simscorer, mode, facial_vector_size=2622):
    """
    Train a classifier while retaining the original scaler and features of the input data, so it can be used in LIME explanations later.

    Parameters
    ----------
    X: numpy array
        Array of size n x 2623: the VGG-Face feature vector and one cell with a list of the HPO IDs
    y: numpy array
        The labels (usually 0 for control and 1 for patient)
    simscorer: object of SimScorer class
        Instance of class for semantic similarity calculations
    mode: str
        Whether to use facial features, HPO data, or both
    facial_vector_size: int
        Size of the feature vector of facial recognition module

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
    X_processed = X[:, :]
    if mode != 'face':
        hpo_terms_pt = X[y == 1, -1]
        hpo_terms_cont = X[y == 0, -1]

        sim_mat = simscorer.calc_full_sim_mat(X)

        sim_avg_pat = sim_mat[:, y == 1].mean(axis=1).reshape(-1, 1)
        sim_avg_control = sim_mat[:, y == 0].mean(axis=1).reshape(-1, 1)

        hpo_features = np.append(sim_avg_pat, sim_avg_control, axis=1)
        scale_hpo = StandardScaler()
        hpo_features = scale_hpo.fit_transform(hpo_features)
        preds, clf_hpo = svm_class(hpo_features, y, hpo_features)
    else:
        scale_hpo, hpo_terms_pt, hpo_terms_cont, clf_hpo = None, None, None, None

    if mode != 'hpo':
        face_features = np.array(X[:, :facial_vector_size], dtype=float)

        scale_face = StandardScaler()
        face_features = normalize(scale_face.fit_transform(face_features))

        preds, clf_face = svm_class(face_features, y, face_features)

        vgg_face_pt = face_features[y == 1, :]
        vgg_face_cont = face_features[y == 0, :]
    else:
        scale_face, vgg_face_pt, vgg_face_cont, clf_face = None, None, None, None

    if mode == 'face':
        clf = clf_face
    elif mode == 'hpo':
        clf = clf_hpo
    elif mode == 'both':
        X = np.append(face_features, hpo_features, axis=1)
        preds, clf = svm_class(X, y, X)
    return clf, hpo_terms_pt, hpo_terms_cont, scale_face, scale_hpo, vgg_face_pt, vgg_face_cont, X_processed, \
           clf_face, clf_hpo


def get_loss(X, y, simscorer, mode, sim_mat):
    """
    Get the predictions for current (possibly randomized) y and X

    Parameters
    ----------
    X: numpy array
        Array of size n x 2623 of the original patients and controls of the suspected syndrome: the VGG-Face feature vector and one cell with a list of the HPO IDs
    y: numpy array
        The y labels
    simscorer: object of SimScorer class
        Instance of class for semantic similarity calculations
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

    if len(X) < 20:
        skf = LeaveOneOut()
        skf.get_n_splits(X, y)
    else:
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(X, y)

    y_pred, y_real, y_ind = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        if mode != 'face':
            resnik_avg_train, resnik_avg_test = simscorer.calc_sim_scores(sim_mat, train_index, test_index, y_train)
            if mode == 'both':
                X_face_train = np.array(X_train[:, :-1], dtype=float)
                X_face_test = np.array(X_test[:, :-1], dtype=float)

            scale = StandardScaler()
            X_hpo_train_norm = scale.fit_transform(resnik_avg_train)
            X_hpo_test_norm = scale.transform(resnik_avg_test)
        else:
            X_face_train = np.array(X_train[:, :-1], dtype=float)
            X_face_test = np.array(X_test[:, :-1], dtype=float)

        if mode != 'hpo':
            scale = StandardScaler()
            X_face_train_norm = normalize(scale.fit_transform(X_face_train))
            X_face_test_norm = normalize(scale.transform(X_face_test))

        if mode == 'face':
            predictions, clf = svm_class(X_face_train_norm, y_train, X_face_test_norm)
        elif mode == 'hpo':
            predictions, clf = svm_class(X_hpo_train_norm, y_train, X_hpo_test_norm)
        elif mode == 'both':
            X_train = np.append(X_face_train_norm, X_hpo_train_norm, axis=1)
            X_test = np.append(X_face_test_norm, X_hpo_test_norm, axis=1)
            predictions, clf = svm_class(X_train, y_train, X_test)

        y_pred.extend(predictions)
        y_real.extend(y_test)
        y_ind.extend(test_index)

    y_real, y_pred, y_ind = np.array(y_real), np.array(y_pred), np.array(y_ind)
    return y_real, y_pred, y_ind


def svm_class(X_train, y_train, X_test):
    """
    Train a support vector machine classifier, with the size of cross-validation for the GridSearch dependent on the size of the training dataset, as described in the paper
    
    Parameters
    ----------
    X_train: numpy array
        Training data with size n x 2624: the VGG-Face feature vector + average HPO similarity for patients + average HPO similarity for controls
    y_train: numpy array
        The labels for the training dataset (usually 0 for control and 1 for patient) 
    X_test: numpy array
        Test data with size n x 2624: the VGG-Face feature vector + average HPO similarity for patients + average HPO similarity for controls
   
    Returns
    -------
    predictions: numpy array
        The prediction score between 0 and 1 for the test data
    clf: sklearn instance
        The trained support vector machine
    """
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV, LeaveOneOut
    from sklearn.calibration import CalibratedClassifierCV

    if len(X_train) < 20:
        zeros_count = len(y_train) - np.count_nonzero(y_train)
        min_count = min(zeros_count, len(y_train) - zeros_count)
        if min_count < 4:
            clf = CalibratedClassifierCV(svm.SVC(), cv=LeaveOneOut())
        else:
            if min_count > 5:
                skf_cv = 5
            else:
                skf_cv = 3
            param_grid = {'estimator__C': [1e-5, 1e-3, 1, 1e3, 1e5]}
            clf = GridSearchCV(
                CalibratedClassifierCV(svm.SVC(), cv=skf_cv), param_grid, cv=skf_cv, n_jobs=-1, scoring='neg_brier_score'
            )
    else:
        param_grid = {'C': [1e-5, 1e-3, 1, 1e3, 1e5]}
        clf = GridSearchCV(
            svm.SVC(probability=True), param_grid, cv=5, n_jobs=-1, scoring='neg_brier_score'
        )
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)[:, 1]
    return predictions, clf
