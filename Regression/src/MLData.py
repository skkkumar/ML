'''
Created on 14-Jan-2018

'''
from DataLoader import DataLoader
import numpy as np
from tabulate import tabulate
import random


class MLData:
    """
    This class deals with ML data
    """
    
    def __init__(self):
        
        self.loader = DataLoader()
        #type of ML problem
        #TYPE = 1 classification, maintain unique classes
        self.TYPE = -1
        
        #number of data sets
        self.SETS = -1
        
        #features
        self.X = []
        #classes 
        self.y = None
        #class names
        self.uniqueClasses = []
    
        
    def loadData(self, pathname, TYPE, SETS):
        """
        This loads ML data
        SETS = 1 (only one sort of data)
        SETS = 2 (training and test data)
        """
        
        self.TYPE = TYPE
        self.SETS = SETS
        
        #check if classification or regression
        if TYPE == 1:
            data, self.uniqueClasses = self.loader.loadDataset(pathname, TYPE)
        elif TYPE == 2:
            data = self.loader.loadDataset(pathname, TYPE)
            
        data = np.asarray(data)
        
        #features
        self.X = np.asarray(data[:,  : -1], dtype = np.float32)
        #classes
        self.y = np.asarray(data[:, -1], dtype = np.uint8)
        
        #normalize last feature
        self.y = np.divide(self.y, self.X[:,0])
        
        #normalize first feature
        min1 = np.amin(self.X[:,0])
        max1 = np.amax(self.X[:,0])
        self.X[:,0] = (self.X[:,0] - min1)/(max1 - min1)        
        
        if SETS == 2:
            
            #shuffle the data
            indexes = np.arange(0, len(self.y))
            random.shuffle(indexes)
            self.X = self.X[indexes, :]
            self.y = self.y[indexes]

            #make first 80% as training data and rest as test data
            first80Count = int(len(self.y) * 8.0 /10.0)
            self.trainX = self.X[0 : first80Count, :]
            self.testX = self.X[first80Count : , :]
            self.trainy = self.y[0 : first80Count]
            self.testy = self.y[first80Count :]
            
 
    
if __name__ == "__main__":
    data = MLData()
    data.loadData("../data/mobile.csv", 2, 1)
    print (tabulate(data.X))
    print (data.y)
