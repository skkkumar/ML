'''
Created on 12 Mar 2018


'''
from MLData import MLData
from SVR import SVR
from Performance import Results
from Visualization import Visualization
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    data = MLData()
    data.loadData("../data/mobile.csv", 2, 2)
    
    kernelTypes = [1, 2, 3, 4]
    trainErrors = []
    testErrors = []
    
    for kType in kernelTypes:
    
        #fit the classifier to the data
        svrModel = SVR(kType)
        svrModel.fitRegressor(data)
        
        #find accuracy
        results = Results()
        trainError, testError = results.calculateError(svrModel, data)
        trainErrors.append(trainError)
        testErrors.append(testError)
        
        #draw line
        vis = Visualization()
        vis.visualizeSVR(data, svrModel)
        
    
    plt.figure()
    plt.plot(kernelTypes, trainErrors, label = "training error")
    plt.plot(kernelTypes, testErrors, label = "test error")
    plt.title('kernel type versus error')
    plt.xlabel('kernel type')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
