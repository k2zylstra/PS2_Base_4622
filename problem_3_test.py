import numpy as np
import matplotlib.pylab as plt
import tests
import data
from sklearn.tree import DecisionTreeClassifier

house_prices = data.HousePrices()
#Workspace 2.1
#TODO: Discretize y_train and y_test
#BEGIN 
for t in [house_prices.y_train, house_prices.y_test]:
    for i in range(len(t)):
        if t[i] < 125000:
            t[i] = 0
        elif t[i] >= 125000 and t[i] < 160000:
            t[i] = 1
        elif t[i] >= 160000 and t[i] < 200000:
            t[i] = 2
        elif t[i] >= 200000:
            t[i] = 3 
#END

def cross_validate(classifier, X, y, train_indices, valid_indices):
    """
    Train classifier on training set and validate on the validation set
    :param classifier: the classifier to use
    :param X: all data of shape (num_samples, num_features)
    :param y: all labels of shape (num_samples)
    :param train_indices:  indices to be used for training the model
    :param valid_indices:  indices to be used for validating the model
    :return: he accuracy of the classifier on the validation set
    """
    valid_accuracy = 0
    #Workspace 3.1
    #TODO: train and validate the model based on provided indices
    #Hint: use score method of the classifier
    #BEGIN 
    dt = classifier
    dt.fit(list(map(X.__getitem__, train_indices)), list(map(y.__getitem__, train_indices)))
    valid_accuracy = dt.score(list(map(X.__getitem__, valid_indices)), list(map(y.__getitem__, valid_indices)))
    #END
    return valid_accuracy


def partition_to_k(permutation, k):
    """
    Partition permutation (shuffled indices) to k different chunks and generate the train/valid splits for the k-fold
    :param permutation: shuffles indices
    :param k: number of folds
    :return: iterable of different k partitions
    """
    size = int(np.ceil(len(permutation)/k))
    for i in range(0, len(permutation), size):
        # valid indices j for which i < o(i) < i + size
        valid_indices = np.where(np.logical_and(permutation>i, permutation<= i + size))[0]
        # train indices j for which o(j) <= j  or o(j) >= j + size
        train_indices = np.where(~np.logical_and(permutation>i, permutation<= i + size))[0]
        yield train_indices, valid_indices

def k_fold_cv(classifier, k, X, y):
    """
    This function performs k-fold cross validation
    :param classifier: a classifier to be used
    :param k: number of folds
    :param X: all training data of shape (num_samples, num_features)
    :param y: all labels of shape (num_samples)
    :return: the average accuracy of the classifier in k-runs
    """
    #shuffle data indices
    permutation = np.random.RandomState(seed=42).permutation(range(X.shape[0]))
    mean_accuracy = 0
    #Workspace 3.3
    #BEGIN 
    for t in partition_to_k(permutation, k):
        mean_accuracy += cross_validate(classifier, X, y, t[0], t[1])
    mean_accuracy += k
    #END
    return mean_accuracy

np.random.seed(4)  # changing the seed might yield different results
best_depth, best_accuracy = -1, 0

#Workspace 3.4
#TODO: 
#BEGIN 
acc_l = []
for d in range(1,11):
    dt = DecisionTreeClassifier(max_features=None, max_depth=d, criterion="entropy", min_samples_split=2)
    acc = k_fold_cv(dt, 5, house_prices.X_train, house_prices.y_train)
    acc_l.append([d, acc])
for i in range(len(acc_l)):
    if acc_l[i][1] > best_accuracy:
        best_accuracy = acc_l[i][1]
        best_depth = acc_l[i][0]
#END
print("Cross validation accuracy for chosen best max_depth %d: %f" % (best_depth, best_accuracy))