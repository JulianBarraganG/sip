from typing import Any 
from numpy.typing import NDArray
import pandas as pd 
import numpy as np 
from pathlib import Path 
from const import CSV_FOLDER, OUTPUT_FOLDER 
from plotting import closed_curves 
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score

def loadData(filename:Path): 
    X = pd.read_csv(filename, usecols=list(range(2,66))).values 
    Y = pd.read_csv(filename, usecols=[1]).values.flatten() 
    return X, Y 

def translate_to_zero(data: NDArray[Any]) -> NDArray[Any]: 
    x_center = np.mean(data[0::2]) 
    y_center = np.mean(data[1::2]) 
    
    data[0::2] -= x_center 
    data[1::2] -= y_center 
    return data 

def scale(S1: NDArray[Any], S2: NDArray[Any]) -> NDArray[Any]: 
    enum = 0 
    denom = 0 

    for n in range(S1.shape[1]): 
        enum += np.inner(S2[:,n], S1[:,n]) 
        denom += np.inner(S2[:,n], S2[:,n]) 
    assert denom !=0, "denominator must not be 0" 
    return enum/denom * S2 

def rotate(S1: NDArray[Any], S2: NDArray[Any]) -> NDArray[Any]: 
    U, _, Vt = np.linalg.svd(S2 @ S1.T) 
    R = Vt.T @ U.T 
    return R @ S2

def train_classifier(X_train, Y_train): 
    clf = svm.SVC(kernel='linear') 
    clf.fit(X_train, Y_train) 
    return clf 

def procrustes_alignment(data_sets: NDArray[Any], target: NDArray) -> NDArray[Any]: 
    target = translate_to_zero(target)
    for m in range(len(data_sets)): 
        data_sets[m] = translate_to_zero(data_sets[m])
        data_sets[m] = scale(target, data_sets[m])
        data_sets[m] = rotate(target, data_sets[m]) 

    return data_sets 


def test_classifier(clf, X_test, Y_test): 
    preds = clf.predict(X_test) 
    return accuracy_score(Y_test, preds)    
    

if __name__ == "__main__":  
    NUM_WINGS = 10 
    DIMS = 2 
    X, Y = loadData(CSV_FOLDER / "BioScan_dataset_Train.csv")  

    X_test, Y_test = loadData(CSV_FOLDER / "BioScan_dataset_Test.csv") 

    outpath = OUTPUT_FOLDER / "task2" 
    outpath.mkdir(parents=False, exist_ok=True) 
    S = np.zeros((X.shape[0], DIMS, X.shape[1]//DIMS))

    S[:,0,:] = X[:,0::2] 
    S[:,1,:] = X[:,1::2]

    closed_curves(S[:10], outpath / "unaligned_wings.png" )
    aligned_S = procrustes_alignment(S, S[0])   

    closed_curves(aligned_S[:10], outpath / "aligned_wings.png") 

    unaligned_svc = train_classifier(X, Y)
    X_al = np.zeros_like(X).astype(np.float64)
    X_al[:,0::2] = aligned_S[:,0,:] 
    X_al[:,1::2] = aligned_S[:,1,:] 
    aligned_svc = train_classifier(X_al, Y) 
    
    print(f"Train score of unaligned classifier is: {test_classifier(unaligned_svc, X, Y)}")
    print(f"Test score of unaligned classifier is: {test_classifier(unaligned_svc, X_test, Y_test)}")

    
    S_tst = np.zeros((X_test.shape[0], DIMS, X_test.shape[1]//DIMS))

    S_tst[:,0,:] = X_test[:,0::2] 
    S_tst[:,1,:] = X_test[:,1::2]

    al_S_tst = procrustes_alignment(S_tst, S[0])

    X_tst_al = np.zeros_like(X_test).astype(np.float64)
    X_tst_al[:,0::2] = al_S_tst[:,0,:]
    X_tst_al[:,1::2] = al_S_tst[:,1,:] 

    print(f"Train score of aligned classifier is: {test_classifier(aligned_svc, X_al, Y)}")
    print(f"Test score of aligned classifier is: {test_classifier(aligned_svc, X_tst_al, Y_test)}")
