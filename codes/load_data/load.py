import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def returnTrainDataSet():
    trainData=np.zeros((50000,1024))
    trainLabels=np.zeros((50000,1024))
    nb_classes=10
    for i in range(1,6):
        t = unpickle('F:\Beheshti\9798-2\DL\HW1\DL_01\cifar10\cifar10\data_batch_'+str(i))
        newt=t[b'data'].reshape((10000,3,1024))
        newtt=np.mean(newt,1)
        newttLabels=np.array(t[b'labels'])
        newttLabels_oneHot = np.eye(nb_classes)[newttLabels]

        if (i==1):
            trainData=newtt
            trainLabels=newttLabels_oneHot
        else:
            trainData=np.append(trainData,newtt,axis=0)
            trainLabels=np.append(trainLabels,newttLabels_oneHot,axis=0)


    return trainData,trainLabels
'''
for i in range(0,10000):
    newtt[i]=np.mean(newt,0)
'''

def returnTestDataSet():
    #testData=np.zeros((10000,1024))
    #testLabels=np.zeros((10000,1024))
    nb_classes=10
    t = unpickle('F:\Beheshti\9798-2\DL\HW1\DL_01\cifar10\cifar10\cifartest_batch')
    newt=t[b'data'].reshape((10000,3,1024))
    newtt=np.mean(newt,1)
    newttLabels=np.array(t[b'labels'])
    newttLabels_oneHot = np.eye(nb_classes)[newttLabels]

    testData=newtt
    testLabels=newttLabels_oneHot


    return testData,testLabels

