'''
Created on 14-Jan-2018

'''
class RegressionModel:
    """
    This is base regressor class
    """
    def __init__(self):
        self.regressor = None
    

    def fitRegressor(self, data):
        """
        This method trains regressor to data
        """
        if data.SETS == 2:
            self.regressor.fit(data.trainX, data.trainy)
    
    
    def predictValue(self, sampleData):
        """
        This method predicts the regression value
        """
        value = self.regressor.predict([sampleData])
        return value[0]
