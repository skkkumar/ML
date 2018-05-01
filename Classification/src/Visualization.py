'''
Created on 14-Jan-2018


'''

import matplotlib.pyplot as plt
import numpy as np
from MLData import MLData


class Visualization:
    """
    This class visualizes the data
    """
    
    
    def visualizeData(self, data, coef = []):
        """
        This method creates 2d plot for the first two features
        """
        fig = plt.figure()
        
        colors = ["red", "blue", "green"]
        
        if data.TYPE == 1:
            for classIndex in range(len(data.uniqueClasses)):
                classData = data.X[data.y == classIndex]
                plt.scatter(classData[:, 0], classData[:, 1], \
                    c=colors[classIndex], edgecolor='k', \
                    label = data.uniqueClasses[classIndex])
                
        if len(coef) != 0:
            minX = np.amin(data.X[:,0])
            maxX = np.amax(data.X[:,0])
            y1 = -(coef[0][0] * minX + coef[0][1])
            y2 = -(coef[0][0] * maxX + coef[0][1])
            plt.plot([minX, maxX], [y1, y2])

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.title("Visualization")
        plt.legend()
        plt.show()
    
    
if __name__ == "__main__":
    
    
    data = MLData()
    data.loadData("../data/fish.csv", 1, 1)
    vis = Visualization()
    vis.visualizeData(data)
