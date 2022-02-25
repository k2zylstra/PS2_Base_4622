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

def split_values(feature_values):
    """
    Helper function to return the split values. if feature consists of the values f1 < f2 < f3 then
    this returns [(f2 + f1)/2, (f3 + f2)/2]
    :param feature_values: 1-d array of shape (num_points)
    :return: array of shape (max(m-1, 1),) where m is the number of unique values in feature_values
    """
    unique_values = np.unique(feature_values)
    if unique_values.shape[0] == 1:
        return unique_values
    return (unique_values[1:] + unique_values[:-1]) / 2


def best_partition(X, y, impurity_measure=entropy):
    """
    :param X: features array, shape (num_samples, num_features)
    :param y: labels of instances in X, shape (num_samples)
    :param impurity_measure: function that takes 1d-array of labels and returns the impurity measure
    :return: Return the best value and its corresponding threshold by splitting based on the different features.
    """

    best_feature, best_threshold, best_reduction = 0, 0, -np.inf

    #Workspace 1.5
    #TODO: Complete the function as detailed in the question and return description
    #BEGIN 
    info_gains = []
    for i in range(X.shape[1]):
        splits = split_values(X[:, i])
        
        ig_max = 0
        best_thresh = 0
        for k in range(len(splits)):
            left_indicies = []
            right_indicies = []
            for j in range(X.shape[0]):
                if X[j][i] <= splits[k]:
                    left_indicies.append(j)
                else:
                    right_indicies.append(j)
            ig = impurity_reduction(y, left_indicies, right_indicies)
            if ig > ig_max:
                ig_max = ig
                best_thresh = splits[k]
            
        info_gains.append((i, ig_max, best_thresh))
    best_split = [0, 0, 0]
    for i in range(len(info_gains)):
        if best_split[1] < info_gains[i][1]:
            best_split = info_gains[i]
    
    best_feature = best_split[0]
    best_threshold = best_split[2]
    best_reduction = best_reduction = best_split[1]
    #END
    return best_feature, best_threshold, best_reduction

class ParentNode:

    def __init__(self, feature_id, feature_threshold, left_child, right_child):
        """
        Initialize a parent node.
        :param feature_id: the feature index on which the splitting will be done
        :param feature_threshold: the feature threshold. Left child takes item with features[features_id] < threshold
        :param left_child: left child node
        :param right_child: right child node
        """
        self.feature_id = feature_id
        self.threshold = feature_threshold
        self.left_child = left_child
        self.right_child = right_child

    def predict(self, x):
        """
        Predict the label of row x. If we're a leaf node, return the value of the leaf. Otherwise, call predict
        of the left/right child (depending on x[feature_index).
        This will be called by DecisionTree.predict
        :param x: 1-d array of shape (num_features)
        :return: integer, the label for x
        """
        if x[self.feature_id] < self.threshold:
            label = self.left_child.predict(x)
        else:
            label = self.right_child.predict(x)
        return label


class DecisionTree:

    def __init__(self, max_depth=-1, min_samples_split=2, impurity_measure=entropy):
        """
        Initialize the decision tree
        :param max_depth: maximum depth of the tree
        :param min_samples_split: minimum number of samples required for a split
        :param impurity_measure: impurity measure function to use for best_partition, default to entropy
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_measure = impurity_measure
        self.root = None
        self.num_features = None

    def build(self, X, y, depth):
        """
        Recursive method used to build the decision tree nodes
        :param X: data that are used to build the tree, of shape (num_samples, num_features)
        :param y: labels of the samples in features, of shape (num_samples)
        :param depth: depth of the tree to create
        :return: the root node of the tree
        """
       
       
        if depth == 0 or len(y) < self.min_samples_split:
            # we reached the maximum depth or we don't have more than the minimum number of samples in the leaf
            return LeafNode(y)
        else:
            # Get the feature, threshold and information_gain of the best split
            feature_id, threshold, gain = best_partition(X, y, self.impurity_measure)
            # gain = 0 occurs when the labels have the same distribution in the child nodes
            # which means that the entropy of the children is the same as the parent's
            if gain > 0:
                # Workspace 1.6
                # TODO: create the left and right child nodes with depth - 1, return the parent node
                #BEGIN 
                # feature id is also index
                left_values = []
                right_values = []
                y_left = []
                y_right = []
                for i in range(X.shape[0]):
                    if X[i][feature_id] < threshold:
                        left_values.append(X[i])
                        y_left.append(y[i])
                    else:
                        right_values.append(X[i])
                        y_right.append(y[i])
                
                left_values = np.array(left_values)
                right_values = np.array(right_values)
                y_left = np.array(y_left)
                y_right = np.array(y_right)
            
                root = ParentNode(feature_id, threshold, self.build(left_values, y_left, depth-1), self.build(right_values, y_right, depth-1))
                return root
                            
                #END
            else:
                # We reach here if information_gain <= 0
                return LeafNode(y)


    def fit(self, X, y):
        """
        :param X: Training samples
        :param y: training labels
        :return: trained classifier
        """
        self.num_features = X.shape[1]
        self.root = self.build(X, y, self.max_depth)
        return self

    def predict(self, X):
        """
        Loops through rows of X and predicts the labels one row at a time
        """
        y_hat = np.zeros((X.shape[0],), int)
        for i in range(X.shape[0]):
            y_hat[i] = self.root.predict(X[i])
        return y_hat

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        :param X: Test samples, shape (num_points, num_features)
        :param y: true labels for X, shape (num_points,)
        :return: mean accuracy
        """
        accuracy = 0
        # Workspace 1.7
        #BEGIN 
        # code here
        #END
        return accuracy