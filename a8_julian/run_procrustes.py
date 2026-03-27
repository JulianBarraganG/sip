import polars as pl
import numpy as np

if __name__ == "__main__":
    from const import DATA_FOLDER, OUTPUT_FOLDER
    from plotting import plot_wings
    from procrustes import procrustes

    NUM_WINGS = 10
    DIMS = 2

    train_path = DATA_FOLDER / "BioSCAN_dataset_Train.csv"
    test_path = DATA_FOLDER / "BioSCAN_dataset_Test.csv"

    X =  pl.read_csv(train_path, columns=range(2, 66))
    X_test = pl.read_csv(test_path, columns=range(2, 66))
    Y =  pl.read_csv(train_path, columns=[1])
    Y_test = pl.read_csv(test_path, columns=[1])

    result_path = OUTPUT_FOLDER / f"{NUM_WINGS}_wings.png"

    assert X.width / DIMS % 2 == 0, "Number of coordinates must correspond"

    # Construxt M x d x N array for procrustes input
    S = np.zeros((X.height, DIMS, X.width // DIMS))
    S[:,0,:] = X[:, 0::2]
    S[:,1,:] = X[:, 1::2]

    aligned_wings = procrustes(S, S[0]) # From train data
    plot_wings(coords=X, aligned=aligned_wings, num_wings=NUM_WINGS, save_path=result_path)

    #### Classifier
    # Make SVM and train on both aligned and unaligned data
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    # Unaligned
    X_unaligned_train, X_unaligned_test = X.to_numpy(), X_test.to_numpy()
    Y_train, Y_test = Y.to_numpy().ravel(), Y_test.to_numpy().ravel()
    svm_unaligned = SVC(kernel="linear", random_state=42)
    svm_unaligned.fit(X_unaligned_train, Y_train)
    preds = svm_unaligned.predict(X_unaligned_test)

    # Check classification performance
    print(f"Unaligned Accuracy: {accuracy_score(Y_test, preds):.4f}")

    # First get aligned test data with same target as train data
    M, d, N = aligned_wings.shape
    assert d == DIMS, "Dimensions must correspond"
    X_aligned_train = aligned_wings.reshape(M, DIMS*N)
    # Aligned X_test
    assert N % 2 == 0, "Number of coordinates must correspond"
    M, N = X_test.shape
    ST = np.zeros((M, DIMS, N // DIMS))
    ST[:,0,:] = X_test[:, 0::2]
    ST[:,1,:] = X_test[:, 1::2]
    X_aligned_test = procrustes(ST, S[0])
    M, d, N = X_aligned_test.shape
    X_aligned_test = X_aligned_test.reshape(M, DIMS*N)

    # Train SVM
    svm_aligned = SVC(kernel="linear", random_state=42)
    svm_aligned.fit(X_aligned_train, Y_train)
    aligned_preds = svm_aligned.predict(X_aligned_test)
    print(f"Aligned Accuracy: {accuracy_score(Y_test, aligned_preds):.4f}")
