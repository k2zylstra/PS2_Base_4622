import numpy as np
import matplotlib.pylab as plt
import tests
import data
from sklearn.tree import DecisionTreeClassifier


features = np.array([
    [37, 44000, 1, 0],
    [61, 52000, 1, 0],
    [23, 44000, 0, 0],
    [39, 38000, 0, 1],
    [48, 49000, 0, 0],
    [57, 92000, 0, 1],
    [38, 41000, 0, 1],
    [27, 35000, 1, 0],
    [23, 26000, 1, 0],
    [38, 45000, 0, 0],
    [32, 50000, 0, 0],
    [25, 52000, 1, 0]
])
labels = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1])

class LeafNode:
    def __init__(self, y):
        """
        :param y: 1-d array containing labels, of shape (num_points,)
        """
        self.label = self.compute_label(y)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.label == other.label)

    @staticmethod
    def compute_label(y):
        """
        return the label that yields best performance if predicted of all instances in y
        :param y:  1-d array containing labels
        :return: single label, integer
        """
        node_label = None
        #Workspace 1.2
        #TODO: Return the label that should be assigned to the leaf node
        #In case of multiple possible labels, choose the one with the lowest value
        #Make no assumptions about the number of class labels
        #BEGIN 
        label_count = {i: 0 for i in np.unique(y)}
        for i in range(len(y)):
            label_count[y[i]] += 1
        max_label = [0,0]
        for l in label_count:
            if label_count[l] > max_label[1]:
                max_label[0] = l
                max_label[1] = label_count[l]
        node_label = max_label[0]
        #END
        return node_label

    def predict(self, x):
        """
        return the label for one obervation x
        :param x: one sample, of shape (num_features)
        :return: label, integer
        """
        return self.label

def entropy(y):
    """
    :param y: 1-d array contains labels, of shape (num_points,)
    :return: float, entropy measure of the labels
    """
    entropy_value = 0
    # Workspace 1.3
    #TODO: Compute the entropy of the labels
    #BEGIN 
    label_count = {i: 0 for i in np.unique(y)}
    for i in range(len(y)):
        label_count[y[i]] += 1
    len_y = len(y)
    for i in label_count:
        p_c = label_count[i]/len_y
        entropy_value += p_c*np.log(p_c)
    entropy_value = -1 * entropy_value
    #END
    return entropy_value

def impurity_reduction(y, left_indices, right_indices, impurity_measure=entropy):
    """
    :param y: all labels
    :param left_indices: the indices of the elements of y that belong to the left child
    :param right_indices: the indices of the elements of y that belong to the right child
    :param impurity_measure: function that takes 1d-array of labels and returns the impurity measure, defaults to entropy
    :return: impurity reduction of the split
    """
    impurity_reduce = 0
    # Workspace 1.4
    #BEGIN 
    M_s = entropy(y)
    P1 = list(map(y.__getitem__, left_indices))
    P2 = list(map(y.__getitem__, right_indices))
    M_sP1 = entropy(P1)
    M_sP2 = entropy(P2)
    car_s = len(y)
    car_P1 = len(P1)
    car_P2 = len(P2)
    
    impurity_reduce = M_s - ( (car_P1/car_s)*M_sP1 + (car_P2/car_s)*M_sP2 )
    
    #END
    return impurity_reduce

tests.test_information_gain(impurity_reduction, entropy)