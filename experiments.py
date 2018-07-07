import csv
import ast
from tqdm import tqdm
from collections import namedtuple, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from itertools import product
import random
import os

def import_data(csv_filename, column_names):
	# column_names is dictionary mapping from the keys to column names in the file
	# expecting keys "fingerprints" and "target"
	with open(csv_filename) as file:
		reader = csv.DictReader(file)
		file = list(reader)
		random.shuffle(file)

		fingerprints = [line[column_names["fingerprints"]] for line in file]
		targets = [line[column_names["target"]] for line in file]

		fingerprints = [ast.literal_eval(fingerprint) for fingerprint in fingerprints]
		targets = [ast.literal_eval(target) for target in targets]

		return (fingerprints, targets)

def classify_evaluate(dataset, classifier, folds):
	"""
		run and evaluate the given classifier on the dataset
	"""

	fingerprints, targets = dataset

	scores = cross_validate(classifier, fingerprints, targets, cv=folds, return_train_score=True, scoring="accuracy")

	test_accuracy = sum(scores["test_score"])/len(scores["test_score"])
	train_accuracy = sum(scores["train_score"]) / len(scores["train_score"])

	return (test_accuracy, train_accuracy)

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
		
	Result = namedtuple("Result", ["accuracy", "train_acc", "classifier", "args"])
	results = []
	for argument in tqdm(arguments):
		classifier = classifier_type(**(argument))

		test_acc, train_acc = classify_evaluate(dataset, classifier, folds)
		result = {"accuracy": test_acc, "train_acc": train_acc, "classifier": classifier_type.__name__}
		result.update(**argument)
		results.append(result)
	
	return results

def random_forest_experiment(dataset):
	classifier = RandomForestClassifier
	classifier_inputs = {
		"max_depth": range(5, 31, 5),
		"n_estimators": range(10, 41, 10),
	}
	folds = 10

	return experiment(dataset, classifier, classifier_inputs, folds)
	

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
	filename = "lxr_nobkg_fingerprints"
	column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}

	dataset = import_data(filename+".csv", column_names)
	
	results = random_forest_experiment(dataset)
	# results = svm_experiment(dataset)

	log_experiment(results, filename+"_pred_results.csv")
	print("max: %s" % (max([(res["accuracy"], res) for res in results])[1],))
