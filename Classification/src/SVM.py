'''
Created on 14-Jan-2018


'''
from ClassifierModel import ClassifierModel
from sklearn import svm
from MLData import MLData
from Performance import Results
from Visualization import Visualization

            
class SVM(ClassifierModel):
    """
    This class deals with support vector machines
    """
    def __init__(self):
        self.classifier = svm.SVC(kernel="linear")
    

if __name__ == "__main__":
    data = MLData()
    data.loadData("../data/fish.csv", 1, 2)
    
    #fit the classifier to the data
    svm = SVM()
    svm.fitClassifier(data)
    
    #find accuracy
    results = Results()
    results.calculateAccuracy(svm, data)
    
    #draw line
    print ("svm line", svm.classifier.coef_)
    vis = Visualization()
    vis.visualizeData(data, svm.classifier.coef_)
