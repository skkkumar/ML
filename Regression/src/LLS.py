'''
Created on 20-Jan-2018


'''
from sklearn.linear_model import LinearRegression
from RegressionModel import RegressionModel
from MLData import MLData
from Visualization import Visualization


class LLS(RegressionModel):
    """
    This class deals with K Nearest Neighbour Classifier
    """
    def __init__(self):
        self.regressor = LinearRegression()
        
        
if __name__ == "__main__":
    
    data = MLData()
    data.loadData("../data/mobile.csv", 2, 2)
    
    #fit the classifier to the data
    lls = LLS()
    lls.fitRegressor(data)
    
    #draw line
    coef = lls.regressor.coef_.flatten().tolist() + [lls.regressor.intercept_]
    print ("LLS line coef", coef)
    vis = Visualization()
    vis.visualizeData(data, coef)
