'''
Created on 20-Jan-2018


'''
from sklearn.neighbors.classification import KNeighborsClassifier
from ClassifierModel import ClassifierModel


class KNN(ClassifierModel):
    """
    This class deals with K Nearest Neighbour Classifier
    """
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=3, \
                                               algorithm='ball_tree')
