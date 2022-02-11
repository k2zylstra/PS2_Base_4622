import os
import pickle
current_folder = os.path.dirname(os.path.abspath(__file__))

class HousePrices(object):
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = pickle.load(open(os.path.join(current_folder, 'test_train.pkl'), 'rb'))

