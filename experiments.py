import csv
import ast
from collections import namedtuple, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from itertools import product
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime
import copy
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

def interpret_score(pred_y, test_y):
    TP = sum([1 for test, pred in zip(test_y, pred_y) if test == pred and pred == 1])
    FP = sum([1 for test, pred in zip(test_y, pred_y) if test != pred and pred == 1])
    TN = sum([1 for test, pred in zip(test_y, pred_y) if test == pred and pred == 0])
    FN = sum([1 for test, pred in zip(test_y, pred_y) if test != pred and pred == 0])
    accuracy = float(TP+TN) / float(TP+FP+TN+FN)
    return {
        "accuracy": accuracy, 
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "MCC": mcc
    }

def my_function(arguments, classifier_type, dataset):
	return results
def sampling(arguments, classifier_type, dataset):
	pos_train_X = []
	pos_train_Y = []
	neg_train_X = []
	neg_train_Y = []

			
	for fingerprint, target in zip(dataset[0], dataset[1]):
		if target == 1:
			pos_train_X.append(fingerprint)
			pos_train_Y.append(target)
		else:
			neg_train_X.append(fingerprint)
			neg_train_Y.append(target)
			
	stop = 0
	step = len(pos_train_X)
	length = len(neg_train_X)
	results = []
	for i in range (length/step):
		train_sample = pos_train_X + neg_train_X, pos_train_Y[stop:stop + step] + neg_train_Y[stop:stop + step]
		stop = stop + step
		results.append(my_function(arguments, classifier_type, train_sample))
		
	result = results[0]
	count = 0
	
	for r in range(1,len(results)):
		for k in results[r]:
			result[k]+=results[r][k]
			count += 1
	for k in result:
		result[k] /= count 
	
	return result
			
			


def cv_layer_2(arguments, classifier_type, dataset, folds):
    fingerprints, targets = dataset
    indexlist = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)

    max_score, max_arg = None, None
    for train_index, test_index in indexlist.split(fingerprints, targets):
        train_X, train_y = np.array(fingerprints)[train_index], np.array(targets)[train_index]
        test_X, test_y = np.array(fingerprints)[test_index], np.array(targets)[test_index]
        for argument in arguments:
            random.seed(datetime.now())
            argument["random_state"] = random.randint(0, 9999999)
            
            sub_dataset = train_X, train_Y

            scores = your_function(arguments, classifier_type, sub_dataset)
            
            
            # classifier = classifier_type(**argument)
            # classifier.fit(train_X, train_y)
            # pred_y = classifier.predict_proba(test_X)
            # score = interpret_score(pred_y, test_y)["accuracy"]

            if max_score == None:
                max_score, max_arg = (score, argument)
            elif max_score <= score:
                max_score, max_arg = (score, argument)
    return max_arg


def cv_layer_1(arguments, classifier_type, dataset, folds):
    fingerprints, targets = dataset

    indexlist = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)

    argument_scores = []
    for train_index, test_index in tqdm(list(indexlist.split(fingerprints, targets))):
        train_X, train_y = np.array(fingerprints)[train_index], np.array(targets)[train_index]
        test_X, test_y = np.array(fingerprints)[test_index], np.array(targets)[test_index]
        dataset_layer_2 = (train_X, train_y)
        
        best_arg = cv_layer_2(arguments, classifier_type, dataset_layer_2, folds)

        classifier = classifier_type(**best_arg)
        classifier.fit(train_X, train_y)
        pred_y = classifier.predict_proba(test_X)
        score = interpret_score(pred_y, test_y)

        argument_scores.append(copy.deepcopy((best_arg, score)))
    
    return argument_scores

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

    time = datetime.now()
    for result in results:
        result["timestamp"] = time

    result_keys_not_in_header = [key for key in results[0].keys() if key not in header]
    header = header + result_keys_not_in_header
    data = data + results

    with open(filename, "wb") as log_file:
        csv_writer = csv.DictWriter(log_file, header, "")
        csv_writer.writeheader()
        for line in data:
            csv_writer.writerow(line)

def experiment(dataset, classifier_type, classifier_inputs, folds, output_log):
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
        arguments.append(classifier_argument)
        
    score_arguments = cv_layer_1(arguments, classifier_type, dataset, folds)
    print("score_arguments %s" % score_arguments)

    results = []
    for score, argument in score_arguments:
        print(score)
        print(argument)
        result = dict(**score)
        result.update(**argument)
        result["classifier"] = classifier_type.__name__

        print(result)
        results.append(result)
    
    print(results)
    log_experiment(results, output_log)

    return score_arguments


def random_forest_experiment(dataset, output_log):
    classifier = RandomForestClassifier
    classifier_inputs = {
        "max_depth": range(1, 82, 20),
        "n_estimators": range(1, 82, 20),
    }
    folds = 10

    return experiment(dataset, classifier, classifier_inputs, folds, output_log)
    

def svm_experiment(dataset, output_log):
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
        results.extend(experiment(dataset, classifier, classifier_inputs, folds, output_log))
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
	
def lxr_experiment():
    input_filename = "lxr_nobkg_fingerprints.csv"
    output_filename = "lxr_nobkg_results.csv"
    column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}

    dataset = import_data(input_filename, column_names)

    random_forest_experiment(dataset, output_filename)
    # svm_experiment(dataset, output_filename)

def smi_to_csv(pos_file, neg_file, output_file):
    data = []

    with open(pos_file, "r") as pos_handle:
        for line in pos_handle:
            data.append({"smiles": line.split("\t")[0], "target": 1})
    with open(neg_file, "r") as neg_handle:
        for line in neg_handle:
            data.append({"smiles": line.split("\t")[0], "target": 0})

    header = ["smiles", "target"]

    with open(output_file, "w+") as output_handle:
        csv_writer = csv.DictWriter(output_handle, header, "")
        csv_writer.writeheader()
        csv_writer.writerows(data)

def make_folder(foldername):
    import os
    try:
        os.makedirs(foldername)
    except:
        pass
def make_files(filenames):
    for filename in filenames:
        with open(filename, "a+"):
            pass

def remove_unkekulizable(csv_file):
    data = []
    headers = []
    with open(csv_file, "rb") as file:
        reader = csv.reader(file)
        headers = reader.next()
        file.seek(0)
        reader = csv.DictReader(file)
        for line in reader:
            if MolFromSmiles(line["smiles"]) != None:
                data.append(line)
            else:
                print("removing %s" % line["smiles"])
    
    with open(csv_file, "w+") as file:
        writer = csv.DictWriter(file, headers, "")
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def dud_experiment():
    dud_datasets = ["ace", "ache", "ada", "alr2", "ampc", "ar", "hmga"]
    dud_datasets = ["ace", "ada", "alr2", "ampc", "ar", "hmga"]

    dud_raw_files = [("dud/%s_actives.smi" % dataset, "dud/%s_background.smi" % dataset)
        for dataset in dud_datasets]
    dud_smile_csv_files = ["dud/smiles/%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/smiles")
    make_files(dud_smile_csv_files)
    dud_fingerprint_files = ["dud/fingerprints/%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/fingerprints")
    make_files(dud_fingerprint_files)
    dud_result_files = ["dud/results/%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/results")
    make_files(dud_result_files)

    print("building csv files from raw files")
    for (raw_pos_file, raw_neg_file), csv_file in zip(dud_raw_files, dud_smile_csv_files):
        print("%s, % s -> %s" % (raw_pos_file, raw_neg_file, csv_file))
        smi_to_csv(raw_pos_file, raw_neg_file, csv_file)

    print("removing unkekulizable molecules")
    for csv_file in dud_smile_csv_files:
        remove_unkekulizable(csv_file)

    print("computing fingerprints")
    from compute_fingerprint import main
    for csv_file, fingerprint_filename in zip(dud_smile_csv_files, dud_fingerprint_files):
        print("%s -> %s " % (csv_file, fingerprint_filename))
        task_params = {'target_name': 'target', 'data_file': csv_file}
        with open(csv_file) as file_handle:
            num_molecules = sum([1 for i in file_handle])-1
            assert(num_molecules > 20)
        N_train = num_molecules - 20
        N_val = 20
        train_val_test_split = (N_train, N_val, 0)
        print("main(%s, %s, %s)" %(task_params, train_val_test_split, fingerprint_filename))
        main(task_params, train_val_test_split, fingerprint_filename)

    print("running fingerprint experiments")
    for fingerprint_filename, result_filename in zip(dud_fingerprint_files, dud_result_files):
        print("%s -> %s" % (fingerprint_filename, result_filename))

        column_names = {"fingerprints": "fingerprints", "target": "target"}

        dataset = import_data(fingerprint_filename, column_names)

        random_forest_experiment(dataset, result_filename)
        # svm_experiment(dataset, result_filename)

if __name__ == "__main__":
    # lxr_experiment()	
    dud_experiment()
