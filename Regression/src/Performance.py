'''
Created on 14-Jan-2018

'''
from MLData import MLData
from LLS import LLS


class Results:
    """
    This class deals with calculating regressor's performance
    """
    
    def calculateError(self, model, data):
        """
        This method calculates the accuracy of the model (classifier/regressor)
        """

        if data.SETS == 2: 
            if data.TYPE == 1:
                
                #initialize
                accuracy = 0  
                   
                #calculate the accuracy
                testSamplesCount = data.testX.shape[0]
                for sampleIndex in range(testSamplesCount):
                    label = model.predictClass(data.testX[sampleIndex, :])
                    if label == data.testy[sampleIndex]:
                        accuracy += 1
                accuracy = float(accuracy)/testSamplesCount
                print ("accuracy", accuracy) 
                 
            else:
                #initialize
                meanSqError = 0     
                #calculate the accuracy on training data
                trainSamplesCount = data.trainX.shape[0]
                for sampleIndex in range(trainSamplesCount):
                    value = model.predictValue(data.trainX[sampleIndex, :])
                    meanSqError += (value - data.trainy[sampleIndex]) ** 2
                trainMeanSqError = float(meanSqError)/trainSamplesCount
                print ("Train Mean square Error", trainMeanSqError) 
                
                #calculate the accuracy on test data
                meanSqError = 0     
                testSamplesCount = data.testX.shape[0]
                for sampleIndex in range(testSamplesCount):
                    value = model.predictValue(data.testX[sampleIndex, :])
                    meanSqError += (value - data.testy[sampleIndex]) ** 2
                testMeanSqError = float(meanSqError)/testSamplesCount
                print ("Test Mean square Error", testMeanSqError) 
                
                return trainMeanSqError, testMeanSqError
                
            
if __name__ == "__main__":    
            
    data = MLData()
    data.loadData("../data/mobile.csv", 2, 2)
    
    print (data.X.shape)
    
    #fit the classifier to the data
    lls = LLS()
    lls.fitRegressor(data)
    
    #find accuracy
    results = Results()
    results.calculateError(lls, data)
