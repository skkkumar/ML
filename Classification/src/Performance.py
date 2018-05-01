'''
Created on 14-Jan-2018


'''
from MLData import MLData
from KNN import KNN


class Results:
    """
    This class deals with calculating classifier's performance
    """
    
    def calculateAccuracy(self, classifier, data):
        """
        This method calculates the accuracy of the classifier
        """
        #initialize
        accuracy = 0            
            
        if data.SETS == 2: 
            #calculate the accuracy
            testSamplesCount = data.testX.shape[0]
            for sampleIndex in range(testSamplesCount):
                label = classifier.predictClass(data.testX[sampleIndex, :])
                if label == data.testy[sampleIndex]:
                    accuracy += 1
            accuracy = float(accuracy)/testSamplesCount
            print ("accuracy", accuracy)   
            
        
    
if __name__ == "__main__":
    data = MLData()
    data.loadData("../data/fish.csv", 1, 2)
    knn = KNN()
    #fit the classifier to the data
    knn.fitClassifier(data)
    
    #find accuracy
    results = Results()
    results.calculateAccuracy(knn, data)
