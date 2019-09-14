from __future__ import print_function
from scipy.io import loadmat
from random import sample
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

def nn(X,Y,test):
    #X: training features
    #Y: labels of X
    #test: test features
    testSize = test.shape[0]
    trainSize = X.shape[0]

    #create prediction result

    preds = np.zeros((testSize,1))
    for testIndex in range(testSize):
        shortestDistance = float('inf')
        for trainIndex in range(trainSize):
            distance = np.dot(np.transpose(test[testIndex] - X[trainIndex]),(test[testIndex] - X[trainIndex]))
            distance = distance ** 0.5
            if(shortestDistance > distance):
                shortestDistance = distance
                preds[testIndex][0] = Y[trainIndex]
        #print(preds[testIndex])
    return preds


if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    num_trials = 10

    #plt.imshow(ocr['data'][0].reshape((28, 28)), cmap=cm.gray_r)
    #plt.show()=
    plotX = []
    plotY = []
    plotErrbars = []
    for n in [ 1000, 2000, 4000,8000]:
        test_err = np.zeros(num_trials)
        XTotal = 0
        for trial in range(num_trials):
            sel = random.sample(range(60000),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            print(ocr['testlabels'].shape)
            test_err[trial] = np.mean(preds != ocr['testlabels'])
        plotX.append(n)
        plotY.append(np.mean(test_err))
        plotErrbars.append(np.std(test_err))
        print("%d\t%g\t%g" % (n,np.mean(test_err),np.std(test_err)))
    #plot result
    plt.errorbar(plotX, plotY, plotErrbars)
    plt.xlabel('sample number (n)')
    plt.ylabel('test error rate')
    plt.show()
#1000	0.11509	0.00319858
#2000	0.08926	0.00272404
#4000	0.06781	0.00242794
#8000	0.05525	0.00147054