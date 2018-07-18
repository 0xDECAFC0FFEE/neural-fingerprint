import csv
import ast
from collections import namedtuple, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, matthews_corrcoef, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from itertools import product
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime
import copy
import os
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def import_data(csv_filename, column_names, random_data_filename=None, unique=True, cutoff=True):
    # column_names is dictionary mapping from the keys to column names in the file
    # expecting keys "fingerprints" and "target"
    with open(csv_filename) as file:
        reader = csv.DictReader(file)
        file = list(reader)
        random.shuffle(file)

        if unique:
            file = set([(line[column_names["fingerprints"]], line[column_names["target"]]) for line in file])
            fingerprints = [line[0] for line in file]
            targets = [line[1] for line in file]
        else:
            fingerprints = [line[column_names["fingerprints"]] for line in file]
            targets = [line[column_names["target"]] for line in file]

        fingerprints = [tuple(ast.literal_eval(fingerprint)) for fingerprint in fingerprints]
        targets = [ast.literal_eval(target) for target in targets]

        if cutoff:
            dataset = list(zip(fingerprints, targets))

            pos_dataset = [d for d in dataset if d[1] == 1][:750]
            neg_dataset = [d for d in dataset if d[1] == 0][:750]

            dataset = pos_dataset + neg_dataset
            random.shuffle(dataset)
            
            fingerprints = [d[0] for d in dataset]
            targets = [d[1] for d in dataset]

        return (fingerprints, targets)

def interpret_score(pred_y_proba, test_y, validation_weights=None, pred_y=None, show_roc=False, **other_things_to_log):
    if pred_y == None:
        pred_y = [max([(index, i) for i, index in enumerate(probs)])[1] for probs in pred_y_proba]

    TP = sum([1 for test, pred in zip(test_y, pred_y) if test == pred and pred == 1])
    FP = sum([1 for test, pred in zip(test_y, pred_y) if test != pred and pred == 1])
    TN = sum([1 for test, pred in zip(test_y, pred_y) if test == pred and pred == 0])
    FN = sum([1 for test, pred in zip(test_y, pred_y) if test != pred and pred == 0])
    accuracy = float(TP+TN) / float(TP+FP+TN+FN)
    mcc = matthews_corrcoef(test_y, pred_y)
    
    weighted_log_loss = log_loss(test_y, pred_y_proba, sample_weight=validation_weights)

    f1 = f1_score(test_y, pred_y, sample_weight=validation_weights)

    pred_max_y_proba = [i[1] for i in pred_y_proba]
    ranking = zip(pred_max_y_proba, pred_y, test_y)
    cur_fp = 0
    for cur_index in range(len(ranking)):
        _, cur_pred_y, cur_test_y = ranking[cur_index]
        if cur_pred_y == 1 and cur_test_y == 0:
            cur_fp += 1
        if cur_fp == 50:
            break
    roc_50_break = cur_index + 1

    pred_y_proba_50 = [i[0] for i in ranking[:roc_50_break]]
    test_y_50 = [i[2] for i in ranking[:roc_50_break]]

    fpr_50, tpr_50, _ = roc_curve(test_y_50, pred_y_proba_50)
    roc50 = auc(fpr_50, tpr_50)

    if show_roc:
        fpr, tpr, _ = roc_curve(test_y, pred_max_y_proba)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot([roc_50_break/float(len(pred_y_proba)), roc_50_break/float(len(pred_y_proba))], [0, 1], color='red', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    result = {
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "MCC": mcc,
        "log_loss": weighted_log_loss,
        "f1": f1,
        "target": f1,
        # "roc50": roc50,
    }
    result.update(other_things_to_log)
    return result

def fit_score_classifier(arguments, classifier_type, dataset, validation_weights=None):
    ((train_X, train_y), (test_X, test_y)) = dataset
    
    classifier = classifier_type(**arguments)
    classifier.fit(train_X, train_y)
    pred_y = classifier.predict_proba(test_X)
    score = interpret_score(pred_y, test_y, validation_weights=validation_weights)
    
    return score

def cv_layer_2(arguments, classifier_type, dataset, folds):
    fingerprints, targets = dataset

    indexlist = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)

    max_score, max_arg = None, None
    for train_index, test_index in tqdm(list(indexlist.split(fingerprints, targets)), position=1, leave=False):
        train_X, train_y = np.array(fingerprints)[train_index], np.array(targets)[train_index]
        test_X, test_y = np.array(fingerprints)[test_index], np.array(targets)[test_index]
        for argument in tqdm(arguments, position=2, leave=False):
            random.seed(datetime.now())
            argument["random_state"] = random.randint(0, 9999999)
            
            train_val_dataset = ((train_X, train_y), (test_X, test_y))
            score = fit_score_classifier(argument, classifier_type, train_val_dataset)

            cur_score = score["target"]

            if max_score == None:
                max_score, max_arg = (cur_score, argument)
            elif "n_estimators" in max_arg and "max_depth" in max_arg and "n_estimators" in argument and "max_depth" in argument:
                if max_score < cur_score:
                    max_score, max_arg = (cur_score, argument)
                elif max_score == cur_score:
                    if max_arg["n_estimators"] + max_arg["max_depth"] > argument["n_estimators"] + argument["max_depth"]:
                        max_score, max_arg = (cur_score, argument)
            elif max_score <= cur_score:
                max_score, max_arg = (cur_score, argument)
    return max_arg


def cv_layer_1(arguments, classifier_type, dataset, folds):
    fingerprints, targets = dataset

    indexlist = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)

    argument_scores = []
    for train_index, test_index in tqdm(list(indexlist.split(fingerprints, targets)), position=0, leave=False):
        train_X, train_y = np.array(fingerprints)[train_index], np.array(targets)[train_index]
        test_X, test_y = np.array(fingerprints)[test_index], np.array(targets)[test_index]
        dataset_layer_2 = (train_X, train_y)
        
        best_arg = cv_layer_2(arguments, classifier_type, dataset_layer_2, folds)

        classifier = classifier_type(**best_arg)
        classifier.fit(train_X, train_y)
        pred_y = classifier.predict_proba(test_X)
        
        num_pos_val = sum(test_y)
        num_neg_val = len(test_y) - num_pos_val
        pos_weight, neg_weight = float(num_neg_val) / float(num_pos_val), 1
        validation_weights = [pos_weight if i == 1 else neg_weight for i in test_y]
        
        score = interpret_score(pred_y, test_y, validation_weights=validation_weights, show_roc=True)

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

    results = []
    for score, argument in score_arguments:
        result = dict(**score)
        result.update(**argument)
        result["classifier"] = classifier_type.__name__

        results.append(result)
    
    log_experiment(results, output_log)
    print(results)

    return score_arguments


def random_forest_experiment(dataset, output_log):
    classifier = RandomForestClassifier
    classifier_inputs = {
        "max_depth": range(1, 101, 20),
        "n_estimators": range(1, 101, 20),
        "class_weight": ["balanced_subsample"],
        "n_jobs": [-1]
    }
    folds = 5

    return experiment(dataset, classifier, classifier_inputs, folds, output_log)


def svm_experiment(dataset, output_log):
    classifier = SVC
    classifier_inputs_list = [
        {
            "kernel": ['poly'],
            "degree": range(2, 3),
            "gamma": [.02, .2, .7],
            "C": [.3, .5, 1],
            "coef0": [0, .1, .5],
            "class_weight": ["balanced"],
            "probability": [True]
        }, 
        {
            "kernel": ['rbf'],
            "gamma": [.02, .2, .7],
            "C": [1, .5, .1],
            "class_weight": ["balanced"],
            "probability": [True]
        },
        {
            "kernel": ['sigmoid'],
            "gamma": [.02, .2, .7],
            "C": [1, .5, .1],
            "coef0": [0, .1, .5],
            "class_weight": ["balanced"],
            "probability": [True]
        }, {
            "kernel": ['linear'],
            "C": [1, .5, .1],
            "class_weight": ["balanced"],
            "probability": [True]
        }
    ]
    folds = 5

    print("svm dataset", sum(dataset[1]))

    results = []
    for classifier_inputs in classifier_inputs_list:
        results.extend(experiment(dataset, classifier, classifier_inputs, folds, output_log))
    return results
	
def mlp_experiment(dataset, output_log):
    classifier = MLPClassifier
    classifier_inputs_list = [{"solver": ['lbfgs'], "hidden_layer_sizes": range(10, 101, 10)}]
    folds = 5
    results = []
    for classifier_inputs in classifier_inputs_list:
        results.extend(experiment(dataset, classifier, classifier_inputs, folds, output_log))
    return results
    
def logreg_experiment(dataset, output_log):
    classifier = LogisticRegression
    
    classifier_inputs_list = [{
       "solver": ['newton-cg'],
        "C": [.3, .5, .6]
    },{
        "solver": ['lbfgs'],
        "C": [.3, .5, .6]
    },{
        "solver": ['liblinear'],
        "C": [.3, .5, .6]
    }]
    folds =5
    results = []
    for classifier_inputs in classifier_inputs_list:
        results.extend(experiment(dataset, classifier, classifier_inputs, folds, output_log))
    return results
    
def lxr_experiment():
    input_filename = "lxr_nobkg_fingerprints.csv"
    output_filename = "lxr_nobkg_results.csv"
    column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}
    random_data_filename = "top1000_rf_fingerprints.csv"

    dataset = import_data(input_filename, column_names, random_data_filename=random_data_filename)

    random_forest_experiment(dataset, output_filename)
    # svm_experiment(dataset, output_filename)
    # mlp_experiment(dataset, output_filename)
    # logreg_experiment(dataset, output_filename)

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
    dud_datasets = ["ar"]

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

    # print("building csv files from raw files")
    # for (raw_pos_file, raw_neg_file), csv_file in zip(dud_raw_files, dud_smile_csv_files):
    #     print("%s, % s -> %s" % (raw_pos_file, raw_neg_file, csv_file))
    #     smi_to_csv(raw_pos_file, raw_neg_file, csv_file)

    # print("removing unkekulizable molecules")
    # for csv_file in dud_smile_csv_files:
    #     remove_unkekulizable(csv_file)

    # print("computing fingerprints")
    # from compute_fingerprint import compute_fingerprints
    # for csv_file, fingerprint_filename in zip(dud_smile_csv_files, dud_fingerprint_files):
    #     # print("%s -> %s " % (csv_file, fingerprint_filename))
    #     with open(csv_file) as file_handle:
    #         num_molecules = sum([1 for i in file_handle])-1
    #         assert(num_molecules > 20)
    #     N_train = num_molecules - 20
    #     N_val = 20
    #     train_val_test_split = (N_train, N_val, 0)

    #     compute_fingerprints(train_val_test_split, fingerprint_filename, data_target_column='target', data_file=csv_file)

    print("running fingerprint experiments")
    for fingerprint_filename, result_filename in zip(dud_fingerprint_files, dud_result_files):
        print("%s -> %s" % (fingerprint_filename, result_filename))

        column_names = {"fingerprints": "fingerprints", "target": "target"}

        dataset = import_data(fingerprint_filename, column_names)

        random_forest_experiment(dataset, result_filename)
        # svm_experiment(dataset, result_filename)
        # mlp_experiment(dataset, result_filename)
        # logreg_experiment(dataset, result_filename)

if __name__ == "__main__":

    # top_1000_experiment()
    # lxr_experiment()
    dud_experiment()

