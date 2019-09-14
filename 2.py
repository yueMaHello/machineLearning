from __future__ import print_function
from scipy.io import loadmat
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def nn(X,Y,test):
    #X: training features
    #Y: labels of X
    #test: test features
    #distance = ((test1-x1)^2 + (test2-x2)^2 + (test3-x3)^2+....)^0.5
    testSize = test.shape[0]
    trainSize = X.shape[0]

    #create prediction result

    preds = np.zeros(testSize)
    for testIndex in range(testSize):
        shortestDistance = float('inf')
        for trainIndex in range(trainSize):
            distance = np.dot(np.transpose(test[testIndex] - X[trainIndex]),(test[testIndex] - X[trainIndex]))
            distance = distance ** 0.5
            if(shortestDistance > distance):
                shortestDistance = distance
                preds[testIndex] = Y[trainIndex]
        print(preds[testIndex])
    return preds


if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    num_trials = 10

    #plt.imshow(ocr['data'][0].reshape((28, 28)), cmap=cm.gray_r)
    #plt.show()

    for n in [ 1000, 2000, 4000, 8000 ]:
        test_err = np.zeros(num_trials)
        for trial in range(num_trials):
            sel = sample(range(len(ocr['data'].astype('float'))),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            print(preds)
            test_err[trial] = np.mean(preds != ocr['testlabels'])
        print("%d\t%g\t%g" % (n,np.mean(test_err),np.std(test_err)))