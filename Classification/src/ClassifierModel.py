'''
Created on 14-Jan-2018


'''
class ClassifierModel:
    """
    This is base classifier class
    """
    def __init__(self):
        self.classifier = None
    
    
    def fitClassifier(self, data):
        """
        This method trains classifier to data
        """
        if data.SETS == 2:
            self.classifier.fit(data.trainX, data.trainy)
    
    
    def predictClass(self, sampleData):
        """
        This method predicts the classifier label
        """
        label = self.classifier.predict([sampleData])
        return label[0]
