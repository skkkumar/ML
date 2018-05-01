'''
Created on 14-Jan-2018

'''

import matplotlib.pyplot as plt
import numpy as np
from MLData import MLData
from mpl_toolkits.mplot3d import Axes3D


class Visualization:
    """
    This class visualizes the data
    """
    
    
    def visualizeData(self, data, coef = []):
        """
        This method creates 3d plot for the first three features
        """
        #create 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
                
        #scatter plot
        ax.scatter(data.X[:, 0], data.X[:, 1], data.y, c='r' )
                
        if len(coef) != 0:
            #create arrays containing surface extremities
            minX1 = np.amin(data.X[:,0])
            maxX1 = np.amax(data.X[:,0])
            minX2 = np.amin(data.X[:,1])
            maxX2 = np.amax(data.X[:,1])
            X = [[minX1, maxX1],
                 [minX1, maxX1]]
            X = np.asarray(X)
            Y = [[minX2, minX2],
                 [maxX2, maxX2]]
            Y = np.asarray(Y)
            Z = X * coef[0] + Y * coef[1] + coef[2]

            ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.title("Visualization")
        plt.show()
    
    
    def visualizeSVR(self, data, svr):
        """
        This method creates 3d plot for the first three features
        It also shows the regression hyperplane
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        #scatter plot
        ax.scatter(data.X[:, 0], data.X[:, 1], data.y, c='r' )
        
        #find extremities
        minX1 = np.amin(data.X[:,0])
        maxX1 = np.amax(data.X[:,0])
        minX2 = np.amin(data.X[:,1])
        maxX2 = np.amax(data.X[:,1])
        
        #create grid
        X = np.arange(minX1, maxX1, 0.1)
        Y = np.arange(minX2, maxX2, 1)
        X, Y = np.meshgrid(X, Y)
           
        #fill z
        Z = np.zeros_like(X)
        for index1 in range(X.shape[0]):
            for index2 in range(X.shape[1]):
                x = X[index1, index2]
                y = Y[index1, index2]
                points = np.asarray([x, y])
                Z[index1, index2] = svr.predictValue(points)
                
        #plot the surface
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False) 
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.title("Visualization")
        plt.show()
    
    
if __name__ == "__main__":
    
    
    data = MLData()
    data.loadData("../data/mobile.csv", 2, 1)
    vis = Visualization()
    vis.visualizeData(data)
