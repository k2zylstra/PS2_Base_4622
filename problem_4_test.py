import numpy as np
import matplotlib.pylab as plt
import tests
import data
from sklearn.tree import DecisionTreeClassifier

from time import time
from sklearn.metrics import precision_score
import pandas as pd


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


def get_weak_learner():
    """Return a new instance of out chosen weak learner"""
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1)

class EnsembleTest:
    """
        Test multiple model performance
    """

    def __init__(self, dataset):
        """
        initialize EnsembleTest
        :param data: dataset containing Training and Test sets
        """
        self.dataset = dataset
        self.execution_time = {} # dictionary with key: model name, value: time taken to fit and score the model
        self.metric = {} # dictionary with key: model name, value: accuracy
        self.scores = {}# dictionary with key: model name, value: weighted average precision
        self.score_name = 'Precision(weighted)'
        self.metric_name = 'Mean accuracy'

    def evaluate_model(self, model, name):
        """
        Fit the model using the training data and save the evaluations metrics on the test set
        :param model: classifier to evaluate
        :param name: name of model
        """
        start = time()
        #Workspace 4.1
        #TODO: Fit the model and get the predictions to compute the metric and the score
        #BEGIN
        model.fit(self.dataset.X_train, self.dataset.y_train)
        
        y_pred = model.predict(self.dataset.X_test)
        
        correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == self.dataset.y_test[i]:
                correct += 1
        accuracy = correct/len(self.dataset.y_test)
        
        self.metric[name] = accuracy
        self.scores[name] = precision_score(self.dataset.y_test, y_pred, average="weighted")
        #END
        self.execution_time[name] = time() - start

    def print_result(self):
        """
            print results for all models trained and tested.
        """
        models_cross = pd.DataFrame({
            'Model': list(self.metric.keys()),
            self.score_name: list(self.scores.values()),
            self.metric_name: list(self.metric.values()),
            'Execution time': list(self.execution_time.values())})
        print(models_cross.sort_values(by=self.score_name, ascending=False))

    def plot_metrics(self):
        """
        Plot bar chart, one for each statistic (metric, score, running time)
        """
        fig, axs = plt.subplots(1, 3)
        fig.set_figheight(6), fig.set_figwidth(18)
        p = 0
        for stats, name in zip([self.metric, self.scores, self.execution_time],
                               [self.metric_name, self.score_name, "Elapsed time"]):
            left = [i for i in range(len(stats))]
            height = [stats[key] for key in stats]
            tick_label = [key for key in stats]
            axs[p].set_title(name)
            axs[p].bar(left, height, tick_label=tick_label, width=0.5)
            p += 1
        plt.show()

class BaggingEnsemble(object):

    def __init__(self, n_estimators, sample_ratio=1.0):
        """
        Initialize BaggingEnsemble
        :param n_estimators: number of estimators/weak learner to use
        :param sample_ratio: ratio of the training data to sample
        """
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.estimators = [] # List used in fit method to store the trained estimators

    def sample_data(self, X_train, y_train):
        X_sample, y_sample = None, None
        #Workspace 4.3
        #TODO: sample random subset of size sample_ratio * len(X_train), sampling is with replacement (iid)
        #BEGIN 
        indicies = np.random.choice(len(X_train), int(self.sample_ratio*len(X_train)), replace=False)
        X_sample = list(map(X_train.__getitem__, indicies))
        y_sample = list(map(y_train.__getitem__, indicies))
        #END
        return X_sample, y_sample

    def fit(self, X_train, y_train):
        """
        Train the different estimators on sampled data using provided training samples
        :param X_train: training samples, shape (num_samples, num_features)
        :param y_train: training labels, shape (num_samples)
        :return: self
        """
        np.random.seed(42) # Keep it to get consistent results across runs, you can change the seed value

        for _ in range(self.n_estimators):
            #Workspace 4.4
            #BEGIN 
            dt = get_weak_learner()
            xt, yt = self.sample_data(X_train, y_train)
            dt.fit(xt, yt)
            self.estimators.append(dt)
            #END
        return self
    def predict(self, X_test):
        """
        Predict the labels of test samples
        :param X_test: array of shape (num_points, num_features)
        :return: 1-d array of shape (num_points)
        """
        predicted_proba = 0
        answer = 0
        #Workspace 4.5
        #TODO: go through the trained estimators and accumulate their predicted_proba to get the mostly likely label
        #BEGIN 
        answer = []
        prob = self.estimators[0].predict_proba(X_test)
        for i in range(1, len(self.estimators)):
            prob += self.estimators[i].predict_proba(X_test)
        for i in range(len(prob)):
            answer.append(np.where(prob[i] == max(prob[i]))[0])
        #END
        return answer

# create a handler for ensemble_test, use the created handler for fitting different models.
ensemble_handler = EnsembleTest(house_prices)
##Workspace 4.2
##TODO: Initialize weak learner and evaluate it using evaluate_model
##BEGIN 
dt_weak = get_weak_learner()
ensemble_handler.evaluate_model(dt_weak, "Weak Learner")
##END
#ensemble_handler.print_result()
#
#
#ensemble_handler.evaluate_model(BaggingEnsemble(10, 0.9), 'Bagging')
#ensemble_handler.print_result()
#ensemble_handler.plot_metrics()

class RandomForest(object):

    def __init__(self, n_estimators, sample_ratio=1.0, features_ratio=1.0):
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.features_ratio = features_ratio
        self.estimators = [] # to store the estimator
        self.features_indices = [] # to store the feature indices used by each estimator

    def sample_data(self, X_train, y_train):
        X_sample, y_sample, features_indices = None, None, None
        #Workspace 4.6
        #TODO: sample random subset of size sample_ratio * len(X_train) and subset of features of size
        #         features_ratio * num_features
        #BEGIN 
        indicies = np.random.choice(len(X_train), int(self.sample_ratio*len(X_train)), replace=True)
        X_sample = list(map(X_train.__getitem__, indicies))
        y_sample = list(map(y_train.__getitem__, indicies))
        
        features_indices = np.random.choice(len(X_train[0]), int(self.features_ratio*len(X_train[0])), replace=False)
        
        to_remove = []
        for i in range(len(X_sample[0])):
            if i not in features_indices:
                to_remove.append(i)
        X_sample = np.delete(X_sample, to_remove, axis=1)
                
        #END
        return X_sample, y_sample, features_indices

    def fit(self, X_train, y_train):
        np.random.seed(42) # keep to have consistent results across run, you can change the value
        for _ in range(self.n_estimators):
            #Workspace 4.7
            #TODO: sample data with random subset of rows and features using sample_data
            #Hint: keep track of the features indices in features_indices to use in predict
            #BEGIN 
            dt = get_weak_learner()
            xt, yt, features = self.sample_data(X_train, y_train)
            dt.fit(xt, yt)
            self.estimators.append(dt)
            self.features_indices.append(features)
            #END

    def predict(self, X_test):
        predicted_proba = 0
        answer = 0
        #Workspace 4.8
        #TODO: compute cumulative sum of predict proba from estimators and return the labels with highest likelihood
        #BEGIN 
        
        answer = []
        

        features_indices = self.features_indices[0]
        X_sample = X_test
        to_remove = []
        for i in range(len(X_sample[0])):
            if i not in features_indices:
                to_remove.append(i)
        X_sample = np.delete(X_sample, to_remove, axis=1)

        prob = self.estimators[0].predict_proba(X_sample)
        for i in range(1, len(self.estimators)):

            features_indices = self.features_indices[0]
            X_sample = X_test.copy()
            to_remove = []
            for i in range(len(X_sample[0])):
                if i not in features_indices:
                    to_remove.append(i)
            X_sample = np.delete(X_sample, to_remove, axis=1)

            prob += self.estimators[i].predict_proba(X_sample)

        for i in range(len(prob)):
            answer.append(np.where(prob[i] == max(prob[i]))[0])
        #END
        return answer

ensemble_handler.evaluate_model(RandomForest(200, sample_ratio=0.7, features_ratio=0.1), 'RandomForest')
ensemble_handler.print_result()