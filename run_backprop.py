from sklearn import datasets, preprocessing, model_selection
from random import seed
from backprop import *
import numpy as np

seed(415)

#Data cleaning
#Prep and re-format the data for use in the networks
#Do train-test splits

def run(network, train, test):
    network.train(train)
    [network.test(p, ['test', 'train'][int(p in train)]) for p in train + test]

def preprocess_data(raw_data):
    data_scaled = preprocessing.scale(raw_data['data'])
    data_names = list(raw_data['target_names'][raw_data['target']])
    data = [(list(data_scaled[i]), data_names[i]) for i in range(len(data_names))]
    data_train, data_test = model_selection.train_test_split(data)
    return data_train, data_test, raw_data['target_names']

#IRIS
iris_train, iris_test, iris_names = preprocess_data(datasets.load_iris())

#WINE
wine_train, wine_test, wine_names = preprocess_data(datasets.load_wine())

#BREAST CANCER
cancer_train, cancer_test, cancer_names = preprocess_data(datasets.load_breast_cancer())

#SPIRAL
N = 300 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype=int) # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

spiral_data = {'target_names': np.array(['class_0', 'class_1', 'class_2']),
               'data': X, 'target': y}
spiral_train, spiral_test, spiral_names = preprocess_data(spiral_data)

#DIGITS
digits_train, digits_test, digits_names = preprocess_data(datasets.load_digits())


#Backprop parameters to record

variables = ['eta', 'mu', 'epochs', 'bias_nodes', 'label', 'data_split', 'prediction']

#IRIS
bp_iris = BackpropNetwork((4, 32, 3), iris_names, .01, .03,
                          NetworkLogger('backprop_sf_iris.csv', variables, 'final_results/'))
bp_iris.connect()
bp_iris.add_bias_units()

run(bp_iris, iris_train, iris_test)

#WINE
bp_wine = BackpropNetwork((13, 32, 3), wine_names, .01, .03,
                          NetworkLogger('backprop_sf_wine.csv', variables, 'final_results/'))
bp_wine.connect()
bp_wine.add_bias_units()

run(bp_wine, wine_train, wine_test)

#CANCER
bp_cancer = BackpropNetwork((30, 64, 2), cancer_names, .01, .03,
                          NetworkLogger('backprop_sf_cancer.csv', variables, 'final_results/'))

bp_cancer.connect()
bp_cancer.add_bias_units()

run(bp_cancer, cancer_train, cancer_test)

#SPIRAL
bp_spiral = BackpropNetwork((2, 100, 3), spiral_names, .01, .03,
                          NetworkLogger('backprop_sf_spiral.csv', variables, 'final_results/'))
bp_spiral.connect()
bp_spiral.add_bias_units()

run(bp_spiral, spiral_train, spiral_test)

#DIGITS
bp_digits = BackpropNetwork((64, 170, 10), digits_names, .01, .03,
                          NetworkLogger('backprop_sf_digits.csv', variables, 'final_results/'))

bp_digits.connect()
bp_digits.add_bias_units()

run(bp_digits, digits_train, digits_test)

