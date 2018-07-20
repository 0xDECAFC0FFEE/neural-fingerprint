import csv
import ast
from collections import namedtuple, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, matthews_corrcoef, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from itertools import product
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime
import copy
import os
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from pprint import pprint
from sklearn.metrics import roc_curve, auc

roc = []
def import_data(csv_filename, column_names, random_data_filename=None, unique=True, cutoff=True):
    """
    column_names is dictionary mapping from the keys to column names in the file. 
    expecting keys "fingerprints" and "target" in column_names
    random data filename is a file of random false fingerprints. if not none, will add 1000 random negative examples.
    if unique = true, importer enforces every returned fingerprint and target is unique
    if cutoff = true, importer cuts off # of negative fingerprint examples imported to 750
    """
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

        fingerprints = [ast.literal_eval(fingerprint) for fingerprint in fingerprints]
        targets = [ast.literal_eval(target) for target in targets]

        if random_data_filename:
            with open(random_data_filename) as random_data:
                reader = csv.DictReader(random_data)
                all_random_fingerprints = [
                    line["fingerprints"] for line in reader]
            random.shuffle(all_random_fingerprints)
            random_fingerprints = list(all_random_fingerprints[:1000])
            fingerprints.extend(random_fingerprints)
            targets.extend([0 for _ in range(len(random_fingerprints))])

        if cutoff:
            dataset = list(zip(fingerprints, targets))

            pos_dataset = [d for d in dataset if d[1] == 1][:750]
            neg_dataset = [d for d in dataset if d[1] == 0][:750]

            dataset = pos_dataset + neg_dataset
            random.shuffle(dataset)
            
            fingerprints = [d[0] for d in dataset]
            targets = [d[1] for d in dataset]

        return (fingerprints, targets)


def show_roc_plot(test_y, pred_y_proba, roc_50_break):
    pred_max_y_proba = [i[1] for i in pred_y_proba]
    fpr, tpr, _ = roc_curve(test_y, pred_max_y_proba)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    break_location = roc_50_break / float(len(pred_y_proba))
    plt.plot([break_location, break_location], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def compute_validation_weights(test_y):
    num_pos_val = sum(test_y)
    num_neg_val = len(test_y) - num_pos_val
    pos_weight, neg_weight = float(num_neg_val) / float(num_pos_val), 1
    validation_weights = [pos_weight if i == 1 else neg_weight for i in test_y]
    return validation_weights

def compute_roc50(pred_y_proba, pred_y, test_y):
    pred_max_y_proba = [i[1] for i in pred_y_proba]
    ranking = zip(pred_max_y_proba, pred_y, test_y)
    sorted(ranking)
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
    return roc50, roc_50_break

def interpret_score(pred_y_proba, test_y, validation_weights=None, pred_y=None, show_roc=False):
    if pred_y == None:
        pred_y = [max([(index, i) for i, index in enumerate(probs)])[1] for probs in pred_y_proba]
    if validation_weights == None:
        validation_weights = compute_validation_weights(test_y)

    TP = sum([1 for test, pred in zip(test_y, pred_y) if test == pred and pred == 1])
    FP = sum([1 for test, pred in zip(test_y, pred_y) if test != pred and pred == 1])
    TN = sum([1 for test, pred in zip(test_y, pred_y) if test == pred and pred == 0])
    FN = sum([1 for test, pred in zip(test_y, pred_y) if test != pred and pred == 0])
    accuracy = float(TP+TN) / float(TP+FP+TN+FN)
    mcc = matthews_corrcoef(test_y, pred_y)
    
    weighted_log_loss = log_loss(test_y, pred_y_proba, sample_weight=validation_weights)

    f1 = f1_score(test_y, pred_y, sample_weight=validation_weights)

    roc50, roc50_break = compute_roc50(pred_y_proba, pred_y, test_y)
    
    result =  {
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "MCC": mcc,
        "log_loss": weighted_log_loss,
        "f1": f1,
        "target":f1,
        "roc50": roc50,
    }

    if show_roc:
        print(result)
        show_roc_plot(test_y, pred_y_proba, roc50_break)

    return result


def bagging(dataset, run, clf_type, arg):

    pos_train = []
    neg_train = []

    f1_s=[]
    for fingerprint, target in zip(dataset[0], dataset[1]):
        if target == 1:
            pos_train.append((fingerprint, target))
        else:
            neg_train.append((fingerprint, target))
    for i in range(run):
        random.shuffle(pos_train)
        random.shuffle(neg_train)
        pos = zip(*(pos_train[:40]))
        neg = zip(*(neg_train[:40]))
        clf = clf_type(**arg)
        clf.fit(pos[0]+neg[0], pos[1]+neg[1])
        pos = zip(*(pos_train[40:60]))
        neg = zip(*(neg_train[40:60]))

        f1_s.append(metrics.f1_score(clf.predict(pos[0]+neg[0]) ,pos[1]+neg[1]))

    return f1_s


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
    '''
    for fingerprint, target in zip(training_dataset[0], training_dataset[1]):
        if target == 1:
            pos_train_X.append(fingerprint)
            pos_train_Y.append(target)
        else:
            neg_train_X.append(fingerprint)
            neg_train_Y.append(target)
    pos = len(pos_train_X)
    neg = len(neg_train_X)
    
    if neg/pos >= 2:
        stop = 0
        results = []
        for i in range (neg/pos):
            train_sample = pos_train_X + neg_train_X[stop:stop + pos], pos_train_Y + neg_train_Y[stop:stop + pos]
            stop = stop + pos
            dataset = (train_sample, validation_dataset)
            results.append(fit_score_classifier(arguments, classifier_type, dataset, validation_weights))
    elif pos/neg >= 2:
        stop = 0
        results = []
        for i in range (pos/neg):
            train_sample = pos_train_X[stop:stop + neg] + neg_train_X, pos_train_Y[stop:stop + neg] + neg_train_Y
            stop = stop + neg
            dataset = (train_sample, validation_dataset)
            results.append(fit_score_classifier(arguments, classifier_type, dataset, validation_weights))
    
    else:
    '''
    dataset = (training_dataset, validation_dataset)
    return fit_score_classifier(arguments, classifier_type, dataset, validation_weights)
        
    result = results[0]
    count = 1
    
    for r in range(1,len(results)):
        for k in results[r]:
            result[k]+=results[r][k]
            count += 1
    for k in result:
        result[k] /= count 
    
    return result

def fit_score_classifier(arguments, classifier_type, dataset, validation_weights=None):
    ((train_X, train_y), (test_X, test_y)) = dataset
    
    classifier = classifier_type(**arguments)
    classifier.fit(train_X, train_y)
    pred_y = classifier.predict_proba(test_X)
    score = interpret_score(pred_y, test_y, validation_weights=validation_weights)
    
    return score


def cv_layer_2(arguments, classifier_type, dataset, folds, sample=False):
    fingerprints, targets = dataset

    skf = StratifiedKFold(n_splits = folds, shuffle = True)

    max_score, max_arg = None, None
    for train_index, test_index in list(skf.split(fingerprints, targets)):
        train_X, test_X = np.array(fingerprints)[train_index], np.array(fingerprints)[test_index]
        train_y, test_y = np.array(targets)[train_index], np.array(targets)[test_index]

        for argument in arguments:
            random.seed(datetime.now())
            argument["random_state"] = random.randint(0, 9999999)
            
            train_val_dataset = ((train_X, train_y), (test_X, test_y))
            
            if sample:
                score = sampling(argument, classifier_type, train_val_dataset)
            else:
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


def cv_layer_1(arguments, classifier_type, dataset, folds, sample=False, bagging=False):
    print("running %s experiment" % classifier_type.__name__)
    fingerprints, targets = dataset

    skf = StratifiedKFold(n_splits = folds, shuffle = True)
    argument_scores = []
    for train_index, test_index in list(skf.split(fingerprints, targets)):
        train_X, test_X = np.array(fingerprints)[train_index], np.array(fingerprints)[test_index]
        train_y, test_y = np.array(targets)[train_index], np.array(targets)[test_index]

        dataset_layer_2 = (train_X, train_y)
        
        best_arg = cv_layer_2(arguments, classifier_type, dataset_layer_2, folds, sample)
        
        if bagging:
            clf = BaggingClassifier(classifier_type(**best_arg))
        else:
            clf = classifier_type(**best_arg)
        clf.fit(train_X, train_y)

        pred_y = clf.predict_proba(test_X)
        
        validation_weights = compute_validation_weights(test_y)
        
        score = interpret_score(pred_y, test_y, validation_weights=validation_weights, show_roc=False)

        argument_scores.append(copy.deepcopy((best_arg, score)))

    return argument_scores, clf

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

def experiment(dataset, classifier_type, classifier_inputs, folds, output_log, sample=False, bagging=False):
    """ 
        generalized experimental setup for the various classifier types
        automatically computes all possible classifier arguments from the ranges given
    """
    print("positives: ", sum(dataset[1]))
    print("negatives: ", len(dataset[1]) - sum(dataset[1]))


    arg_names = classifier_inputs.keys()
    arg_ranges = classifier_inputs.values()

    arguments = []
    for arg_vals in product(*arg_ranges):
        classifier_argument = zip(arg_names, arg_vals)
        classifier_argument = {arg_name: arg_val for arg_name, arg_val in classifier_argument}
        arguments.append(classifier_argument)
    
    score_arguments, clf = cv_layer_1(arguments, classifier_type, dataset, folds, sample, bagging)

    results = []
    for score, argument in score_arguments:
        result = dict(**score)
        result.update(**argument)
        result["classifier"] = classifier_type.__name__

        results.append(result)
    
    print(results)
    log_experiment(results, output_log)

    return score_arguments, clf


def random_forest_experiment(dataset, output_log):
    classifier = RandomForestClassifier
    classifier_inputs = {
        "max_depth": range(1, 101, 20),
        "n_estimators": range(1, 101, 20),
        "class_weight": ["balanced_subsample"],
        "n_jobs": [-1]
    }
    folds = 5

    return experiment(dataset, classifier, classifier_inputs, folds, output_log, sample=False)


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
        results.extend(experiment(dataset, classifier, classifier_inputs, folds, output_log, sample=False))
    return results
	
def mlp_experiment(dataset, output_log, bagging=False):
    classifier = MLPClassifier
    classifier_inputs_list = [{
                "solver": ['lbfgs'],
        "hidden_layer_sizes": [10],
        "alpha": [.0001, .001, .01, .1,10,100]
    }]
    folds = 5
    results = []
    clfs = []
    for classifier_inputs in classifier_inputs_list:
        output, clf = experiment(dataset, classifier, classifier_inputs, folds, output_log, sample=True, bagging=bagging)
        results.extend(output)
        clfs.append(clf)
    return results, clfs
    
def logreg_experiment(dataset, output_log, bagging=False):
    classifier = LogisticRegression
    
    classifier_inputs_list = [{
       "solver": ['newton-cg', 'lbfgs'],
        "C": [.3,.6,.9,1.2],
       "class_weight": ['balanced']
       
    }]
    folds =5
    results = []
    clfs = []
    for classifier_inputs in classifier_inputs_list:
        output, clf = experiment(dataset, classifier, classifier_inputs, folds, output_log, sample=True, bagging=bagging)
        results.extend(output)
        clfs.append(clf)
    return results, clfs
    
def lxr_experiment():
    input_filename = "lxr_nobkg_fingerprints.csv"
    output_filename = "lxr_nobkg_results.csv"
    column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}
    random_data_filename = "top1000_rf_fingerprints.csv"

    dataset = import_data(input_filename, column_names, random_data_filename=random_data_filename)

    # random_forest_experiment(dataset, output_filename)
    # svm_experiment(dataset, output_filename)
    mlp_experiment(dataset, output_filename)
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

def compute_fingerprints(dud_raw_files, dud_smile_csv_files, dud_fingerprint_files):
    print("RECOMPUTING FINGERPRINTS. CANCEL NOW OR WAIT 10 MINUTES")
    print("building csv files from raw files")
    for (raw_pos_file, raw_neg_file), csv_file in zip(dud_raw_files, dud_smile_csv_files):
        print("%s, % s -> %s" % (raw_pos_file, raw_neg_file, csv_file))
        smi_to_csv(raw_pos_file, raw_neg_file, csv_file)

    print("removing unkekulizable molecules")
    for csv_file in dud_smile_csv_files:
        remove_unkekulizable(csv_file)

    print("computing fingerprints")
    from compute_fingerprint import compute_fingerprints
    for csv_file, fingerprint_filename in zip(dud_smile_csv_files, dud_fingerprint_files):
        # print("%s -> %s " % (csv_file, fingerprint_filename))
        with open(csv_file) as file_handle:
            num_molecules = sum([1 for i in file_handle])-1
            assert(num_molecules > 20)
        N_train = num_molecules - 20
        N_val = 20
        train_val_test_split = (N_train, N_val, 0)

        compute_fingerprints(train_val_test_split, fingerprint_filename, data_target_column='target', data_file=csv_file)

def dud_experiment():
    dud_datasets = ["ace", "ache", "ada", "alr2", "ampc", "ar", "hmga"]
    #dud_datasets = ["ace"]
    dud_raw_files = [("dud/%s_actives.smi" % dataset, "dud/%s_background.smi" % dataset)
        for dataset in dud_datasets]
    dud_smile_csv_files = ["dud/smiles/%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/smiles")
    make_files(dud_smile_csv_files)
    dud_fingerprint_files = ["dud/fingerprints/%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/fingerprints")
    make_files(dud_fingerprint_files)
    dud_result_files = ["dud/results/NN_REG_%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/results")
    make_files(dud_result_files)

    # compute_fingerprints(dud_raw_files, dud_smile_csv_files, dud_fingerprint_files)

    for fingerprint_filename, result_filename in zip(dud_fingerprint_files, dud_result_files):
        print("%s -> %s" % (fingerprint_filename, result_filename))

        column_names = {"fingerprints": "fingerprints", "target": "target"}

        dataset = import_data(fingerprint_filename, column_names)

        # random_forest_experiment(dataset, result_filename)
        # svm_experiment(dataset, result_filename)
        mlp_experiment(dataset, result_filename)
        logreg_experiment(dataset, result_filename)

def bagging_experiment(clf, arg, input_filename = "lxr_nobkg_fingerprints.csv"):
    
    column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}
    fingerprints, targets = import_data(input_filename, column_names)
    with open("vecs_zinc.txt") as file:
        file = list(file)
        file = [line for line in file]
        random.shuffle(file)
        count = 0
        for line in file[:1000]:
            fingerprints.append(line.split(" ")[:50])
        targets+[0]*1000
    dataset = (fingerprints, targets)
    
    f1 = bagging(dataset, 50, clf, arg)
    plt.hist(f1, alpha = 0.5)




if __name__ == "__main__":
    '''
    lxr_experiment()
    clf = LogisticRegression
    arg = {"solver" : 'newton-cg', "C" : 0.6}
    bagging_experiment(clf, arg)
    clf = MLPClassifier
    arg = {"solver" : 'lbfgs',
       "hidden_layer_sizes": 30}
    bagging_experiment(clf, arg)
    '''
    # top_1000_experiment()
    # lxr_experiment()
    dud_experiment()

