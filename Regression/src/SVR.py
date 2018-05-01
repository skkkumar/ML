
'''
Created on 20-Jan-2018

'''
from sklearn.svm import SVR as svr
from RegressionModel import RegressionModel
from MLData import MLData
from Performance import Results
from Visualization import Visualization



class SVR(RegressionModel):
    """
    This class deals with K Nearest Neighbour Classifier
    """
    def __init__(self, kernelType, CValue = 1):
        if kernelType == 1:
            self.regressor = svr(kernel='linear', C=CValue)
        elif kernelType > 1:
            self.regressor = svr(kernel='poly', C=CValue, degree=kernelType)
        
        
if __name__ == "__main__":
    data = MLData()
    data.loadData("../data/mobile.csv", 2, 2)
    
    svrModel = SVR(1)
    svrModel.fitRegressor(data)
    
    #find accuracy
    results = Results()
    results.calculateError(svrModel, data)
    
    #draw line
    vis = Visualization()
    vis.visualizeSVR(data, svrModel)
        
