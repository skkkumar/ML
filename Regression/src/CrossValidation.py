'''
Created on 10 Mar 2018


'''
from MLData import MLData
from sklearn.svm import SVR as svr
from Performance import Results
from sklearn.model_selection import GridSearchCV
from SVR import SVR


class CrossValidation:
    """
    This class performs k fold cross validation to find best C of SVR
    """
    
    def findBestModelParameters(self, mldata, parameters, kFoldCount):
        """
        This method uses cross validation to find the best SVR paramter C
        """
        clf = GridSearchCV(svr(), parameters, cv=kFoldCount,
                       scoring='%s' % "neg_mean_squared_error")
        clf.fit(mldata.trainX, mldata.trainy)
    
        print("Best C value :")
        print(clf.best_params_['C'])
        return clf.best_params_['C']
    
    
if __name__ == "__main__":
    
    data = MLData()
    data.loadData("../data/mobile.csv", 2, 2)
    
    #perform cross validation
    parameters = [{'kernel': ['linear'], \
                         'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}]
    crossValidation = CrossValidation()
    C = crossValidation.findBestModelParameters(data, parameters, 3)

    svrModel = SVR(1, C)
    svrModel.fitRegressor(data)
    
    #find accuracy
    results = Results()
    results.calculateError(svrModel, data)
