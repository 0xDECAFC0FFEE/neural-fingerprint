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
import time
from datetime import datetime
import copy
import os
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from pprint import pprint
from sklearn.metrics import roc_curve, roc_auc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcROC
from timeout_decorator import timeout

random.seed(datetime.now())
np.set_printoptions(linewidth=2000)

def import_random_data(random_data_filename, dataset):
    if random_data_filename:
        fingerprints, targets = dataset

        with open(random_data_filename) as random_data:
            reader = csv.DictReader(random_data)
            all_random_fingerprints = [line["fingerprints"] for line in reader]

        random.shuffle(all_random_fingerprints)
        random_fingerprints = list(all_random_fingerprints[:1000])

        fingerprints.extend(random_fingerprints)
        targets.extend([0 for _ in range(len(random_fingerprints))])

        dataset = fingerprints, targets
    return dataset

def import_cutoff(dataset):
    fingerprints, targets = dataset
    dataset = list(zip(fingerprints, targets))

    pos_dataset = [d for d in dataset if d[1] == 1][:750]
    neg_dataset = [d for d in dataset if d[1] == 0][:750]

    dataset = pos_dataset + neg_dataset
    random.shuffle(dataset)

    fingerprints = [d[0] for d in dataset]
    targets = [d[1] for d in dataset]

    return fingerprints, targets

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

        dataset = (fingerprints, targets)
        dataset = import_random_data(random_data_filename, dataset)

        if cutoff:
            dataset = import_cutoff(dataset)

        return dataset


def show_roc_plot(test_y, pred_y_proba, roc_50_break):
    pred_max_y_proba = [i[1] for i in pred_y_proba]
    break_location = roc_50_break / float(len(pred_y_proba))

    fpr, tpr, _ = roc_curve(test_y, pred_max_y_proba)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
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

    roc50 = roc_auc_score(test_y_50, pred_y_proba_50)

    return roc50, roc_50_break

def compute_bedroc(pred_y_proba, test_y):
    bedroc = sorted(zip([i[1] for i in pred_y_proba], test_y))
    bedroc = [[j] for i, j in bedroc]
    bedroc = CalcBEDROC(scores=bedroc, col=0, alpha=20)
    return bedroc

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
    # mcc = matthews_corrcoef(test_y, pred_y)

    weighted_log_loss = log_loss(test_y, pred_y_proba, sample_weight=validation_weights)

    f1 = f1_score(test_y, pred_y, sample_weight=validation_weights)

    roc50, roc50_break = compute_roc50(pred_y_proba, pred_y, test_y)
    roc = roc_auc_score(test_y, [i[1] for i in pred_y_proba])
    bedroc = compute_bedroc(pred_y_proba, test_y)

    result =  {
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TPR": TP/float(TP+FN),
        "TNR": TN/float(TN+FP),
        # "MCC": mcc,
        "log_loss": weighted_log_loss,
        "f1": f1,
        "roc50": roc50,
        "rocNOT50": roc,
        "bedroc": bedroc
    }

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

def fit_score_classifier(arguments, clf_type, dataset, validation_weights=None):
    ((train_X, train_y), (test_X, test_y)) = dataset

    classifier = clf_type(**arguments)
    # print("fitting %s" % str(arguments))
    classifier.fit(train_X, train_y)
    pred_y = classifier.predict_proba(test_X)
    score = interpret_score(pred_y, test_y, validation_weights=validation_weights)

    return score

def max_result_argument(res_arg_1, res_arg_2, target_colname):
    result_1, arg_1 = res_arg_1
    result_2, arg_2 = res_arg_2

    if result_1 == None:
        return res_arg_2
    elif result_2 == None:
        return res_arg_1

    score_1 = result_1[target_colname]
    score_2 = result_2[target_colname]

    if score_1 < score_2:
        return res_arg_2
    elif score_2 < score_1:
        return res_arg_1
    else:
        if "n_estimators" in arg_1 and "max_depth" in arg_1 and "n_estimators" in arg_2 and "max_depth" in arg_2:
            try:
                if arg_1["n_estimators"] + arg_1["max_depth"] > arg_2["n_estimators"] + arg_2["max_depth"]:
                    return res_arg_2
                else:
                    return res_arg_1
            except:
                if arg_1["max_depth"] == None:
                    return res_arg_2
                else:
                    return res_arg_1
        else:
            return res_arg_1


def cv_layer_2(clf_arguments, clf_type, dataset, non_clf_arguments):
    fingerprints, targets = dataset

    skf = StratifiedKFold(n_splits=non_clf_arguments["cv2_folds"], shuffle=True)

    max_res_arg = (None, None)
    for train_index, test_index in tqdm(list(skf.split(fingerprints, targets)), position=1, leave=False):
        train_X, test_X = np.array(fingerprints)[train_index], np.array(fingerprints)[test_index]
        train_y, test_y = np.array(targets)[train_index], np.array(targets)[test_index]


        for clf_argument in tqdm(clf_arguments, position=2, leave=False):
            clf_argument["random_state"] = random.randint(0, 9999999)

            train_val_dataset = ((train_X, train_y), (test_X, test_y))

            if non_clf_arguments["sample"]:
                result = sampling(clf_argument, clf_type, train_val_dataset)
            else:
                result = fit_score_classifier(clf_argument, clf_type, train_val_dataset)

            max_res_arg = max_result_argument(max_res_arg, (result, clf_argument), non_clf_arguments["target"])
    return max_res_arg


def cv_layer_1(clf_arguments, clf_type, dataset, non_clf_arguments):
    print("running %s experiment" % clf_type.__name__)

    fingerprints, targets = dataset

    skf = StratifiedKFold(n_splits=non_clf_arguments["cv1_folds"], shuffle = True)
    for train_index, test_index in tqdm(list(skf.split(fingerprints, targets)), position=0, leave=False):
        train_X, test_X = np.array(fingerprints)[train_index], np.array(fingerprints)[test_index]
        train_y, test_y = np.array(targets)[train_index], np.array(targets)[test_index]

        dataset_layer_2 = (train_X, train_y)

        best_res, best_arg = cv_layer_2(clf_arguments, clf_type, dataset_layer_2, non_clf_arguments)
        if non_clf_arguments["bagging"]:
            clf = BaggingClassifier(clf_type(**best_arg))
        else:
            clf = clf_type(**best_arg)

        clf.fit(train_X, train_y)

        pred_y = clf.predict_proba(test_X)

        validation_weights = compute_validation_weights(test_y)

        score = interpret_score(pred_y, test_y, validation_weights=validation_weights, show_roc=True)

        yield copy.deepcopy((score, best_arg))

def log_experiment(results, filename, default_header=[], overwrite=False, copy_raw=False):
    if overwrite:
        open(filename, "w+").close()
    else:
        open(filename, "a+").close()
    with open(filename, "r+b") as log_file:
        try:
            csv_reader = csv.reader(log_file)
            header = next(csv_reader)
            log_file.seek(0)
            data = list(csv.DictReader(log_file))
        except:
            header = []
            data = []

    if not copy_raw:
        for result in results:
            result["timestamp"] = time.strftime("%I:%M %m-%d")
            try:
                result["batch_num"] = log_experiment.batch_number
            except:
                log_experiment.batch_number = random.randint(0, 9999)
                result["batch_num"] = log_experiment.batch_number

    keys_to_add = set(results[0].keys() + default_header) if results else set(default_header)
    result_keys_not_in_header = [key for key in keys_to_add if key not in header]
    result_keys_not_in_header = sorted(result_keys_not_in_header)

    header = header + result_keys_not_in_header
    data = data + results

    with open(filename, "wb") as log_file:
        csv_writer = csv.DictWriter(log_file, header, "")
        csv_writer.writeheader()
        for line in data:
            csv_writer.writerow(line)

def experiment(dataset, clf_type, clf_args_config, non_clf_arguments, output_log):

    """
        generalized experimental setup for the various classifier types
        automatically computes all possible classifier arguments from the ranges given
    """
    print("positives: ", sum(dataset[1]))
    print("negatives: ", len(dataset[1]) - sum(dataset[1]))


    arg_names = clf_args_config.keys()
    arg_ranges = clf_args_config.values()

    clf_arguments = []
    for arg_vals in product(*arg_ranges):
        classifier_argument = zip(arg_names, arg_vals)
        classifier_argument = {arg_name: arg_val for arg_name, arg_val in classifier_argument}
        clf_arguments.append(classifier_argument)

    score_arguments = cv_layer_1(clf_arguments, clf_type, dataset, non_clf_arguments)

    for testing_score, argument in score_arguments:
        result = dict(**testing_score)
        result.update(**argument)
        result["classifier"] = clf_type.__name__
        result["classifier_arguments"] = argument
        result.update(**non_clf_arguments)
        log_experiment([result], output_log)

        print("\n\n %s \n" % result)

    return score_arguments


def random_forest_experiment(dataset, output_log):
    classifier = RandomForestClassifier
    clf_args_config = {
        "max_depth": range(20, 161, 20) + [None],
        "n_estimators": range(20, 161, 20),
        "class_weight": ["balanced_subsample"],
        "n_jobs": [-1],
        "criterion": ["gini", "entropy"],
    }
    non_clf_arguments = {
        "cv1_folds": 5,
        "cv2_folds": 5,
        "sample": False,
        "bagging": False,
        "target": "f1",
    }

    return experiment(dataset, classifier, clf_args_config, non_clf_arguments, output_log)


def svm_experiment(dataset, output_log):
    classifier = SVC
    clf_args_config_list = [
        {
            "kernel": ['poly'],
            "degree": range(2, 3),
            "C": [1, .5, .3],
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
        # {
        #     "kernel": ['sigmoid'],
        #     "gamma": [.02, .2, .7],
        #     "C": [1, .5, .1],
        #     "coef0": [0, .1, .5],
        #     "class_weight": ["balanced"],
        #     "probability": [True]
        # },
        # {
        #     "kernel": ['linear'],
        #     # "C": [1, .5, .1],
        #     "C": [1],
        #     "class_weight": ["balanced"],
        #     "probability": [True]
        #     "cache_size": [1000]
        # }
    ]
    non_clf_arguments = {
        "cv1_folds": 5,
        "cv2_folds": 5,
        "sample": False,
        "bagging": False,
        "target": "f1",
    }

    results = []
    for clf_args_config in clf_args_config_list:
        results.extend(experiment(dataset, classifier, clf_args_config, non_clf_arguments, output_log))
    return results

def mlp_experiment(dataset, output_log, bagging=False):
    classifier = MLPClassifier
    clf_args_config_list = [{
                "solver": ['lbfgs'],
        "hidden_layer_sizes": [10],
        "alpha": [.0001, .001, .01, .1,10,100]
    }]
    non_clf_arguments = {
        "cv1_folds": 5,
        "cv2_folds": 5,
        "sample": True,
        "bagging": bagging,
        "target": "roc50",
    }
    results = []

    for clf_args_config in clf_args_config_list:
        output = experiment(dataset, classifier, clf_args_config, non_clf_arguments, output_log)

        results.extend(output)

    return results

def logreg_experiment(dataset, output_log, bagging=False):
    classifier = LogisticRegression

    clf_args_config_list = [{
       "solver": ['newton-cg', 'lbfgs'],
        "C": [.3,.6,.9,1.2],
       "class_weight": ['balanced']
    }]
    non_clf_arguments = {
        "cv1_folds": 5,
        "cv2_folds": 5,
        "sample": True,
        "bagging": bagging,
        "target": "roc50",
    }
    results = []

    for clf_args_config in clf_args_config_list:
        output = experiment(dataset, classifier, clf_args_config, non_clf_arguments, output_log)

        results.extend(output)
    return results

def compile_experiment_results(input_files, target, batch_num=None):
    """
    compiles results of experiments by classifier type
    crawls through each dataset, finds the batch number of its most recent run.
    filters for most recent batch and finds best result for each classifier
    """
    results = []
    default_header = set()
    for filename in input_files:
        with open(filename) as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            default_header.update(header)
            file.seek(0)

            csv_reader = list(csv.DictReader(file))

            if not batch_num:
                batch_num = csv_reader[-1]["batch_num"]

            cur_batch = [i for i in csv_reader if i["batch_num"] == str(batch_num)]
            batch_cls = list(set([i["classifier"] for i in cur_batch]))
            best_expr_results = [max([i for i in cur_batch if i["classifier"] == cls], key=lambda a: a[target]) for cls in batch_cls]

            for result in best_expr_results:
                result["filename"] = filename

            results.extend(copy.deepcopy(best_expr_results))

    default_header = sorted(default_header)

    if batch_num:
        output_file = "compiled_experiment_%s_results.csv" % batch_num
    else:
        output_file = "compiled_experiment_results.csv"

    log_experiment(results, output_file, default_header=list(default_header), overwrite=True, copy_raw=True)
    os.system("open %s" % output_file)

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

    compile_experiment_results([output_filename], "f1")

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
        open(filename, "a+").close()

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
    dud_datasets = ["ace", "ache", "alr2", "ampc", "ar"]
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

    # compute_fingerprints(dud_raw_files, dud_smile_csv_files, dud_fingerprint_files)

    for fingerprint_filename, result_filename in zip(dud_fingerprint_files, dud_result_files):
        print("%s -> %s" % (fingerprint_filename, result_filename))

        column_names = {"fingerprints": "fingerprints", "target": "target"}

        dataset = import_data(fingerprint_filename, column_names)

        random_forest_experiment(dataset, result_filename)
        svm_experiment(dataset, result_filename)
        # mlp_experiment(dataset, result_filename)
        # logreg_experiment(dataset, result_filename)

    compile_experiment_results(dud_result_files, "f1")

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
    # lxr_experiment()
    dud_experiment()
    # compile_experiment_results(dud_result_files, "f1")
