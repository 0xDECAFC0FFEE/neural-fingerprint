import csv
import itertools as it
import ast
import numpy as np
from tqdm import tqdm
from collections import namedtuple, defaultdict
from sklearn.model_selection import cross_validate
from itertools import product
import random
import os
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def read_csv(filename, nrows, input_name, target_name):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in it.islice(reader, nrows):
            #print np.fromstring(row[input_name][1:-1], dtype=float, sep = ',').shape
            data[0].append(np.fromstring(row[input_name][1:-1], dtype=float, sep = ','))
            data[1].append(float(row[target_name]))
    return map(np.array, data)

def load_data(filename, sizes, input_name, target_name):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_data_slices_nolist(filename, slices, input_name, target_name)

def load_data_slices_nolist(filename, slices, input_name, target_name):
    stops = [s.stop for s in slices]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)
    return [(data[0][s], data[1][s]) for s in slices]

def classify_evaluate(dataset, classifier, folds):
	"""
		run and evaluate the given classifier on the dataset
	"""

	fingerprints, targets = dataset

	scores = cross_validate(classifier, fingerprints, targets, cv=folds, return_train_score=True, scoring="accuracy")

	test_accuracy = sum(scores["test_score"])/len(scores["test_score"])
	train_accuracy = sum(scores["train_score"]) / len(scores["train_score"])
        dist = scores["test_score"]
	return (test_accuracy, train_accuracy, dist)

def log_experiment(results, filename):
	with open(filename, "a+") as log_file:
		pass
	with open(filename, "r+b") as log_file:
		try:
			csv_reader = csv.reader(log_file)
			header = next(csv_reader)
			log_file.seek(0)
			data = list(csv.DictReader(log_file))
		except:
			header = []
			data = []
	
	result_keys_not_in_header = [key for key in results[0].keys() if key not in header]
	header = header + result_keys_not_in_header
	data = data + results

	with open(filename, "wb") as log_file:
		csv_writer = csv.DictWriter(log_file, header, "")
		csv_writer.writeheader()
		for line in data:
			csv_writer.writerow(line)

def experiment(dataset, classifier_type, classifier_inputs, folds):
	""" 
		generalized experimental setup for the various classifier types
		automatically computes all possible classifier arguments from the ranges given
	"""

	arg_names = classifier_inputs.keys()
	arg_ranges = classifier_inputs.values()

	arguments = []
	for arg_vals in product(*arg_ranges):
		classifier_argument = zip(arg_names, arg_vals)
		classifier_argument = {arg_name: arg_val for arg_name, arg_val in classifier_argument}
		classifier_argument["random_state"] = random.randint(0, 9999)
		arguments.append(classifier_argument)
		
	Result = namedtuple("Result", ["accuracy", "train_acc", "classifier", "distribution", "args"])
	results = []
	for argument in tqdm(arguments):
		classifier = classifier_type(**(argument))

		test_acc, train_acc, dist = classify_evaluate(dataset, classifier, folds)
		result = {"accuracy": test_acc, "train_acc": train_acc, "classifier": classifier_type.__name__, "distribution": np.array(dist)}
		result.update(**argument)
		results.append(result)
                
	return results

def mlp_experiment(dataset):
	classifier = MLPClassifier
	classifier_inputs_list = [{
                "solver": ['lbfgs'],
		"hidden_layer_sizes": range(10, 101, 10)
	},{
                "solver": ['adam'],
		"hidden_layer_sizes": range(10, 101, 10)
	},{
                "solver": ['sgd'],
		"hidden_layer_sizes": range(10, 101, 10)
	}]
	folds = 10
	results = []
	for classifier_inputs in classifier_inputs_list:
		results.extend(experiment(dataset, classifier, classifier_inputs, folds))
	return results
	
def logreg_experiment(dataset):
        classifier = LogisticRegression
	
	classifier_inputs_list = [{
		"solver": ['liblinear'],
		"C": [.3, .5, .6, .7, .9, 1]
	},{
		"solver": ['newton-cg'],
		"C": [.3, .5, .6, .7, .9, 1]
	},{
		"solver": ['lbfgs'],
		"C": [.3, .5, .6, .7, .9, 1]
	},{
		"solver": ['sag'],
		"C": [.3, .5, .6, .7, .9, 1]
	},{
		"solver": ['saga'],
		"C": [.3, .5, .6, .7, .9, 1]
	}]
	folds =10
	results = []
	for classifier_inputs in classifier_inputs_list:
		results.extend(experiment(dataset, classifier, classifier_inputs, folds))
	return results
	
def svm_experiment(dataset):
	classifier = SVC
	folds = 10
	classifier_inputs_list = [{
		"kernel": ['poly'],
		"degree": range(2, 3),
		"gamma": [.001, .01, .1, .5, 1], 
		"C": [.3, .5, .6, .7, .9, 1],
		"coef0": [0, .1, .5],
		"decision_function_shape": ['ovo', 'ovr'],
	}, {
		"kernel": ['rbf'],
		"gamma": [.001, .01, .1, .5, 1],
		"C": [.3, .5, .6, .7, .9, 1],
		"decision_function_shape": ['ovo', 'ovr'],
	}, {
		"kernel": ['sigmoid'],
		"gamma": [.001, .01, .1, .5, 1],
		"C": [.3, .5, .6, .7, .9, 1],
		"coef0": [0, .1, .5],
		"decision_function_shape": ['ovo', 'ovr'],
	}, {
		"kernel": ['linear'],
		"C": [.3, .5, .6, .7, .9, 1],
		"decision_function_shape": ['ovo', 'ovr'],
	}]

	results = []
	for classifier_inputs in classifier_inputs_list:
		results.extend(experiment(dataset, classifier, classifier_inputs, folds))
	return results

if __name__ == "__main__":
	task_params = {'target_name': 'LXRbeta binder',
               'data_file': 'lxr_nobkg_fingerprints.csv'}
	N_train = 140
	N_val = 0
	N_test = 0

	print "Loading data..."
	traindata, valdata, testdata = load_data(
			task_params['data_file'], (N_train, N_val, N_test),
			input_name='fingerprints', target_name=task_params['target_name'])
	train_inputs, train_targets = traindata
	val_inputs,   val_targets   = valdata
	test_inputs,  test_targets  = testdata
	dataset = (train_inputs, train_targets)
	results = mlp_experiment(dataset)
	log_experiment(results, "mlp_pred.csv")
	results = logreg_experiment(dataset)
	log_experiment(results, "logreg_pred.csv")
	print("max: %s" % (max([(res["accuracy"], res) for res in results])[1],))
