import sys
sys.path.append('./neuralfingerprint')

from neuralfingerprint import *
from import_output import *
from scoring import *
from dud_data_preparation import *
from compute_fingerprint import compute_fingerprints

import random
from datetime import datetime
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
import itertools
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from tqdm import tqdm
import copy
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import sklearn

random.seed(datetime.now())
np.set_printoptions(linewidth=2000000)


def sampling(arguments, classifier_type, dataset):
    pos_train_X = []
    pos_train_Y = []
    neg_train_X = []
    neg_train_Y = []

    training_dataset, validation_dataset = dataset
    test_y = validation_dataset[1]
    num_pos_val = sum(test_y)
    num_neg_val = len(test_y) - num_pos_val
    pos_weight, neg_weight = float(num_neg_val)/float(num_pos_val),1

    validation_weights = [pos_weight if i == 1 else neg_weight for i in test_y]
    dataset = (training_dataset, validation_dataset)
    return fit_score_classifier(arguments, classifier_type, dataset, validation_weights=validation_weights)

    result = results[0]
    count = 1

    for r in range(1,len(results)):
        for k in results[r]:
            result[k]+=results[r][k]
            count += 1
    for k in result:
        result[k] /= count

    return result

def fit_score_classifier(arguments, clf_type, dataset, validation_weights=None,  non_clf_arguments={"bagging": False}):
    ((train_X, train_y), (test_X, test_y)) = dataset

    classifier = clf_type(**arguments)

    if non_clf_arguments["bagging"]:
        classifier = BaggingClassifier(classifier)

    classifier.fit(train_X, train_y)
    pred_y = classifier.predict_proba(test_X)
    score = interpret_score(pred_y, test_y, validation_weights=validation_weights)

    return score

def cv_layer_2(clf_arguments, clf_type, dataset, non_clf_arguments):
    fingerprints, targets = dataset

    skf = StratifiedKFold(n_splits=non_clf_arguments["cv2_folds"], shuffle=True)

    max_res_arg = (None, None)
    for train_index, test_index in tqdm(list(skf.split(fingerprints, targets))):
        train_X, test_X = np.array(fingerprints)[train_index], np.array(fingerprints)[test_index]
        train_y, test_y = np.array(targets)[train_index], np.array(targets)[test_index]


        for clf_argument in clf_arguments:
            clf_argument["random_state"] = random.randint(0, 9999999)

            train_val_dataset = ((train_X, train_y), (test_X, test_y))

            if non_clf_arguments["sample"]:
                result = sampling(clf_argument, clf_type, train_val_dataset)
            else:
                result = fit_score_classifier(clf_argument, clf_type, train_val_dataset, non_clf_arguments=non_clf_arguments)

            max_res_arg = max_result_argument(max_res_arg, (result, clf_argument), non_clf_arguments["target"])
    return max_res_arg


def cv_layer_1(clf_arguments, clf_type, datasets, non_clf_arguments):
    print("running %s experiment" % clf_type.__name__)

    for (cv1_train, cv1_test) in datasets:
        train_X, train_y = cv1_train
        test_X, test_y = cv1_test

        dataset_layer_2 = (train_X, train_y)

        best_res, best_arg = cv_layer_2(clf_arguments, clf_type, dataset_layer_2, non_clf_arguments)
        
        clf = clf_type(**best_arg)

        if non_clf_arguments["bagging"]:
            clf = BaggingClassifier(clf)

        clf.fit(train_X, train_y)

        pred_y = clf.predict_proba(test_X)

        validation_weights = compute_validation_weights(test_y)

        score = interpret_score(pred_y, test_y, validation_weights=validation_weights, show_roc=True)

        yield copy.deepcopy((score, best_arg))



def experiment(datasets, clf_type, clf_args_config, non_clf_arguments, output_log):
    """
        generalized experimental setup for the various classifier types
        automatically computes all possible classifier arguments from the ranges given
    """

    arg_names = clf_args_config.keys()
    arg_ranges = clf_args_config.values()

    clf_arguments = []
    for arg_vals in itertools.product(*arg_ranges):
        classifier_argument = zip(arg_names, arg_vals)
        classifier_argument = {arg_name: arg_val for arg_name, arg_val in classifier_argument}
        clf_arguments.append(classifier_argument)

    score_arguments = cv_layer_1(clf_arguments, clf_type, datasets, non_clf_arguments)

    for testing_score, argument in score_arguments:
        result = dict(**testing_score)
        result.update(**argument)
        result["classifier"] = clf_type.__name__
        result["classifier_arguments"] = argument
        result.update(**non_clf_arguments)
        log_experiment([result], output_log)

        print("\n\n %s \n" % result)

    return score_arguments


def random_forest_experiment(datasets, output_log, cv1_folds):
    classifier = RandomForestClassifier
    clf_args_config = {
        "max_depth": range(20, 100, 20),
        "n_estimators": range(20, 100, 20),
        "class_weight": ["balanced_subsample"],
        "n_jobs": [-1],
        "criterion": ["gini", "entropy"],
    }
    non_clf_arguments = {
        "cv1_folds": cv1_folds,
        "cv2_folds": 5,
        "sample": False,
        "bagging": False,
        "target": "roc",
    }

    return experiment(datasets, classifier, clf_args_config, non_clf_arguments, output_log)


def svm_experiment(dataset, output_log, cv1_folds):
    classifier = SVC
    clf_args_config_list = [
        {
            "kernel": ['poly'],
            "degree": range(1, 3),
            "C": [1, .5, .3],
            "coef0": [0, .5, .75],
            "class_weight": ["balanced"],
            "probability": [True]
        },
        {
            "kernel": ['rbf'],
            "gamma": [.02, .03, .01],
            "C": [1, .5, .1],
            "class_weight": ["balanced"],
            "probability": [True]
        },
    ]
    non_clf_arguments = {
        "cv1_folds": cv1_folds,
        "cv2_folds": 5,
        "sample": False,
        "bagging": True,
        "target": "roc",
    }

    results = []
    for clf_args_config in clf_args_config_list:
        results.extend(experiment(dataset, classifier, clf_args_config, non_clf_arguments, output_log))
    return results


def mlp_experiment(dataset, output_log, cv1_folds):
    classifier = MLPClassifier
    clf_args_config_list = [{
        "solver": ['lbfgs'],
        "hidden_layer_sizes": [10],
        "alpha": [.0001, .001, .01, .1,10,100]
    }]
    non_clf_arguments = {
        "cv1_folds": cv1_folds,
        "cv2_folds": 5,
        "sample": False,
        "bagging": False,
        "target": "roc",
    }
    results = []

    for clf_args_config in clf_args_config_list:
        output = experiment(dataset, classifier, clf_args_config, non_clf_arguments, output_log)

        results.extend(output)

    return results


def logreg_experiment(dataset, output_log, cv1_folds):
    classifier = LogisticRegression

    clf_args_config_list = [{
       "solver": ['newton-cg', 'lbfgs'],
        "C": [.3,.6,.9,1.2],
       "class_weight": ['balanced']
    }]
    non_clf_arguments = {
        "cv1_folds": cv1_folds,
        "cv2_folds": 5,
        "sample": False,
        "bagging": False,
        "target": "roc",
    }
    results = []

    for clf_args_config in clf_args_config_list:
        output = experiment(dataset, classifier, clf_args_config, non_clf_arguments, output_log)

        results.extend(output)
    return results


def lxr_experiment():
	input_filename = "lxr_nobkg_fingerprints.csv"
	output_filename = "lxr_nobkg_results.csv"
	column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}
	random_data_filename = "top1000_rf_fingerprints.csv"

	dataset = import_data(input_filename, column_names, random_data_filename=random_data_filename)

	random_forest_experiment(dataset, output_filename)
	svm_experiment(dataset, output_filename)
	mlp_experiment(dataset, output_filename)
	logreg_experiment(dataset, output_filename)

	compile_experiment_results([output_filename], "f1")

def compute_fingerprints_wrapper(folds, cv1_trains_tests_lrs, dud_parsed_crk3d):
    cv1_trains, cv1_tests, lrs = cv1_trains_tests_lrs
    
    for train_files, test_files, learning_rate, crk3d in zip(cv1_trains, cv1_tests, lrs, dud_parsed_crk3d):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        X, y = crk3d

        for (train_val_indices, test_indices), train_file, test_file in zip(skf.split(X, y), train_files, test_files):
            X_train_val, X_test = np.array(X)[train_val_indices], np.array(X)[test_indices]
            y_train_val, y_test = np.array(y)[train_val_indices], np.array(y)[test_indices]

            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, shuffle=True, test_size=.3)

            # X_train, y_train = X_train_val[:20], y_train_val[:20]
            # X_val, y_val = X_train_val[20:], y_train_val[20:]

            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            
            X_val, y_val = sklearn.utils.shuffle(X_val, y_val)
            

            print("len X_train", len(X_train))
            print("len X_val", len(X_val))
            print("len X_val pos", len([val for val in y_val if val == 1]))

            dataset = (X_train, y_train), (X_val, y_val), (X_test, y_test)

            compute_fingerprints(dataset, train_file, test_file, learning_rate)

def dud_experiment(recompute_csv=True, recompute_fingerprints=True):
    dud_datasets = ["ace", "ache", "alr2", "ampc", "ar"]
    
    learning_rates = defaultdict(lambda:np.exp(-6))
    learning_rates = [learning_rates[dataset] for dataset in dud_datasets]

    dud_raw_smile_files = [("dud/%s_actives.smi" % dataset, "dud/%s_background.smi" % dataset) for dataset in dud_datasets]
    dud_smile_csv = ["dud/smiles/%s.csv" % dataset for dataset in dud_datasets]
    dud_parsed_csv = []
    dud_result_files = make_files(folder="dud/results", filenames=[i + ".csv" for i in dud_datasets])
    
    # for raw, smile_csv_filename in zip(dud_raw_smile_files, dud_smile_csv):
    #     if recompute_csv:
    #         compute_csv_files(raw, smile_csv_filename)
    #     with open(smile_csv_filename) as smile_csv:
    #         dud_parsed_csv.append(read_csv(smile_csv_filename, len(list(csv.DictReader(smile_csv))), "smiles", "target"))

    # cv1_folds=3
    # fingerprint_train_all_dataset, fingerprint_test_all_dataset = [], []

    # for dataset in dud_datasets:
    #     for fold_num in range(cv1_folds):
    #         make_files(folder="dud/fingerprint/"+dataset+"/"+str(fold_num))
    #     fingerprint_train_dataset = make_files(filenames=[("dud/fingerprint/" + dataset + "/cv_" + str(fold_num) + "/train.csv") for fold_num in range(cv1_folds)])
    #     fingerprint_test_dataset = make_files(filenames=[("dud/fingerprint/" + dataset + "/cv_" + str(fold_num) + "/test.csv") for fold_num in range(cv1_folds)])
    #     fingerprint_train_all_dataset.append(fingerprint_train_dataset)
    #     fingerprint_test_all_dataset.append(fingerprint_test_dataset)

    # if recompute_fingerprints:
    #     compute_fingerprints_wrapper(cv1_folds, (fingerprint_train_all_dataset, fingerprint_test_all_dataset, learning_rates), dud_parsed_csv)

    # for fingerprint_train_dataset, fingerprint_test_dataset, result_filename in zip(fingerprint_train_all_dataset, fingerprint_test_all_dataset, dud_result_files):
    #     print("%s -> %s" % (fingerprint_train_dataset, result_filename))

    #     column_names = {"fingerprints": "fingerprints", "target": "target"}

    #     cv1_datasets = []

    #     for fingerprint_train, fingerprint_test in zip(fingerprint_train_dataset, fingerprint_test_dataset):
    #         fingerprint_train = import_data(fingerprint_train, column_names)
    #         fingerprint_test = import_data(fingerprint_test, column_names)
    #         cv1_datasets.append((fingerprint_train, fingerprint_test))

    #     random_forest_experiment(cv1_datasets, result_filename, cv1_folds)
    #     svm_experiment(cv1_datasets, result_filename, cv1_folds)
    #     mlp_experiment(cv1_datasets, result_filename, cv1_folds)
    #     logreg_experiment(cv1_datasets, result_filename, cv1_folds)

    compile_experiment_results(dud_result_files, "roc", default_batch_num=None)

if __name__ == "__main__":
    # lxr_experiment()
    dud_experiment()
