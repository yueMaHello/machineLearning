from __future__ import print_function
from scipy.io import loadmat
import numpy as np
#external library
from sklearn.model_selection import KFold

def compute_distances(X, X_train):
    dists = np.sqrt(-2*np.dot(X, X_train.T) + np.sum(np.square(X_train), axis = 1) + np.transpose([np.sum(np.square(X), axis = 1)]))
    return dists

def nn(X,Y,test):
    #X: training features
    #Y: labels of X
    #test: test features

    #create prediction result
    dists = compute_distances(test, X)
    min_dists_index = np.argmin(dists, axis = 1)
    preds = Y[min_dists_index]
    return preds

def knn_find_best_K(X,Y):
    #X: training features; Y: labels of X; test: test features; k: fold number

    kf = KFold(n_splits = 10, shuffle= True, random_state = 3)
    #record test error sum of 10 k values
    test_err_sum = np.zeros(10)
    #split into 10 fold and run 10 times for each k
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        dists = compute_distances(X_test, X_train)
        for k in range(1,11):
            kPreds = np.zeros((Y_test.shape[0],1)).astype(int)
            for i in range(Y_test.shape[0]):
                dists_sort_i = np.argpartition(dists[i], k)[:k]
                firstKElements = Y_train[dists_sort_i]
                kPreds[i][0] = findMajority(firstKElements)
            test_err_sum[k-1] += np.mean(kPreds != Y_test)
            print('test on k = ', k, test_err_sum)
    test_err_avg = test_err_sum/10
    print('error for k = 1 to 10: ', test_err_avg)
    bestK = np.argmin(test_err_avg) + 1
    print('bestK',bestK)
    return bestK

def knn(X,Y,test, k):
    dists = compute_distances(test, X)
    preds = np.zeros((test.shape[0],1)).astype(int)
    for i in range(test.shape[0]):
        dists_sort_i = np.argpartition(dists[i], k)[:k]
        firstKElements = Y[dists_sort_i]
        preds[i][0] = findMajority(firstKElements)

    return preds

def findMajority(X):
    #print(X.flatten().shape)
    preds = np.bincount(X.flatten()).argmax()
    return preds

if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    kValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    bestK= knn_find_best_K(ocr['data'].astype('float'), ocr['labels'])
    preds = knn(ocr['data'].astype('float'),ocr['labels'],ocr['testdata'].astype('float'), bestK)
    #find index of K having min test_err
    test_err = np.mean(preds != ocr['testlabels'])
    print(test_err)

#error for k = 1 to 10:  [0.02753333 0.03395    0.02741667 0.02873333 0.02831667 0.02978333
# 0.02971667 0.03085    0.0311     0.03245]
#bestK = 3： 0.0295