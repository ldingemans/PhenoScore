import numpy as np

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
    
    param_grid = {'C': [1e-5, 1e-3, 1, 1e3, 1e5]}
    
    if (np.sum(y_train == 0) < 2) or (np.sum(y_train == 1) < 2):
        clf = svm.SVC(probability=True)
    else:
        if len(X_train) < 10:
             skf = LeaveOneOut()
        else:
             skf = 5
        clf = GridSearchCV(
            svm.SVC(probability=True), param_grid, cv=skf,  n_jobs=-1, scoring='neg_brier_score'
            )
        
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)[:,1]
    return predictions, clf    

