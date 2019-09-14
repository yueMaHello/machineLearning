from __future__ import print_function
from scipy.io import loadmat
import numpy as np
#external library
from sklearn.model_selection import KFold

def compute_distances(X, X_train):
    dists = np.sqrt(-2*np.dot(X, X_train.T) + np.sum(np.square(X_train), axis = 1) + np.transpose([np.sum(np.square(X), axis = 1)]))
    return dists
def knn(X,Y,test, kValues):
    #X: training features; Y: labels of X; test: test features; k: fold number

    kf = KFold(n_splits = 10, shuffle= True, random_state = 1)
    #record test error sum of 10 k values
    test_err_sum = np.zeros(10)
    #split into 10 fold and run 10 times for each k
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        dists = compute_distances(X_test, X_train)
        #get sorting index of dists grouped by row
        dists_sort = np.argsort(dists, axis=1)
        for k in range(1,11):
            firstKCandidates = findFirstKElements(Y_train, dists_sort, k)
            kPreds = findMajority(firstKCandidates)
            test_err_sum[k-1] += np.mean(kPreds != Y_test)
            print('test on k = ', k, test_err_sum)
    return test_err_sum/10

def findFirstKElements(Y, Xindeces, k):

    result = np.zeros((Xindeces.shape[0], k))
    for i in range(Xindeces.shape[0]):
        XindecsList = Xindeces[i].tolist()
        for j in range(k):
            result[i][j] = int(Y[XindecsList.index(j)])
    return result.astype(int)

def findMajority(X):
    preds = np.zeros(X.shape[0]).astype(int)
    for i in range(X.shape[0]):
        preds[i] = np.bincount(X[i]).argmax()
    return preds

if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    kValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_err = np.zeros(len(kValues))

    array = np.array([[1,2,3,4,5],[2,3,4,5,1],[3,4,5,1,2]])
    dists_sort = np.argsort(array, axis=1)
    print(dists_sort)
    Y = np.array([[1], [2], [3], [3], [5]])
    print(Y.flatten())
    f3 = findFirstKElements(Y, dists_sort, 3)
    print(f3)
    print(findMajority(f3))
    #
    # print(dists_sort)
    #
    #
    # print('f3', f3)
    # print(findMajority(f3))

