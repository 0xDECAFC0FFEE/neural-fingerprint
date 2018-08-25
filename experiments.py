import csv
import ast
from collections import namedtuple, defaultdict
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import log_loss, matthews_corrcoef, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import itertools
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
from tempfile import TemporaryFile
import os
import re
from rdkit import Chem
from compute_fingerprint import compute_fingerprints

random.seed(datetime.now())
np.set_printoptions(linewidth=2000000)

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

        # print("importing data for file:", csv_filename)
        # print("targets:", targets)
        # print("fingerprints:", fingerprints)


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

def average_scores(scores, target=None):
    average_score = defaultdict(lambda: float(0))
    scores_to_include = {"accuracy", "TP", "TN", "FP", "FN", "TPR", "TNR",
        "log_loss", "f1", "roc50", "rocNOT50", "bedroc", "batch_num"}
    for score in scores:
        for key, value in score.items():
            if key not in scores_to_include:
                continue
            average_score[key] += ast.literal_eval(value)
    for key, value in average_score.items():
        average_score[key] /= len(scores)
    return dict(average_score)


def max_scores(scores, target):
    max_score = 0
    for score in scores:
        if score[target] > max_score:
            max_score = score
    return score

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
                log_experiment.batch_number = random.randint(0, 99999999)
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
        "target": "f1",
    }

    return experiment(datasets, classifier, clf_args_config, non_clf_arguments, output_log)


def svm_experiment(dataset, output_log, cv1_folds):
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
    ]
    non_clf_arguments = {
        "cv1_folds": cv1_folds,
        "cv2_folds": 5,
        "sample": False,
        "bagging": False,
        "target": "f1",
    }

    results = []
    for clf_args_config in clf_args_config_list:
        results.extend(experiment(dataset, classifier, clf_args_config, non_clf_arguments, output_log))
    return results


def mlp_experiment(dataset, output_log, cv1_folds, bagging=False):
    classifier = MLPClassifier
    clf_args_config_list = [{
                "solver": ['lbfgs'],
        "hidden_layer_sizes": [10],
        "alpha": [.0001, .001, .01, .1,10,100]
    }]
    non_clf_arguments = {
        "cv1_folds": cv1_folds,
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


def logreg_experiment(dataset, output_log, cv1_folds, bagging=False):
    classifier = LogisticRegression

    clf_args_config_list = [{
       "solver": ['newton-cg', 'lbfgs'],
        "C": [.3,.6,.9,1.2],
       "class_weight": ['balanced']
    }]
    non_clf_arguments = {
        "cv1_folds": cv1_folds,
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

def compile_experiment_results(input_files, target, default_batch_num=None):
    """
    compiles results of experiments by classifier type
    crawls through each dataset, finds the batch number of its most recent run.
    filters for most recent batch and finds best result for each classifier
    """
    results = []
    default_header = set()
    batch_nums = set()
    for filename in input_files:
        with open(filename) as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            default_header.update(header)
            file.seek(0)

            csv_reader = list(csv.DictReader(file))

            batch_num = csv_reader[-1]["batch_num"] if not default_batch_num else default_batch_num
            batch_nums.add(batch_num)

            cur_batch = [i for i in csv_reader if i["batch_num"] == str(batch_num)]

            batch_cls = list(set([i["classifier"] for i in cur_batch]))

            average_expr_results = [average_scores([i for i in cur_batch if i["classifier"] == clf], target) for clf in batch_cls]

            for result in average_expr_results:
                result["filename"] = filename

            print("average results", average_expr_results)

            results.extend(copy.deepcopy(average_expr_results))

    default_header = sorted(default_header)

    print(len(batch_nums))

    if len(batch_nums) == 1 or default_batch_num:
        batch_num = batch_nums.pop() if not default_batch_num else default_batch_num
        print("batch num: %s" % batch_num)
        output_file = "compiled_experiment_%s_results.csv" % batch_num
    else:
        output_file = "compiled_experiment_results.csv"

    print(results)

    log_experiment(results, output_file, default_header=list(default_header), overwrite=True, copy_raw=True)
    os.system("open %s" % output_file)

# def lxr_experiment():
# 	input_filename = "lxr_nobkg_fingerprints.csv"
# 	output_filename = "lxr_nobkg_results.csv"
# 	column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}
# 	random_data_filename = "top1000_rf_fingerprints.csv"

# 	dataset = import_data(input_filename, column_names, random_data_filename=random_data_filename)

# 	random_forest_experiment(dataset, output_filename)
# 	# svm_experiment(dataset, output_filename)
# 	# mlp_experiment(dataset, output_filename)
# 	# logreg_experiment(dataset, output_filename)

# 	compile_experiment_results([output_filename], "f1")

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
    
def make_files(folder=None, filenames=[]):
    if folder:
        try:
            os.makedirs(folder)
        except:
            pass
        filenames = ["%s%s%s" % (folder, os.sep, fn) for fn in filenames]
        for filename in filenames:
            open(filename, "a+").close()
    else:
        for filename in filenames:
            open(filename, "a+").close()
    
    return filenames

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

class Mol:
    def __init__(self, name):
        self.atoms = []
        self.bonds = defaultdict(lambda: defaultdict(lambda: []))
        self.bond_list = []
        self.name = name
        self.ligand = False
        self.smile = ""

    def distance(self, atom_1_index, atom_2_index):
        atom_1 = self.atoms[atom_1_index]
        atom_2 = self.atoms[atom_2_index]
        return pow(
            pow(float(atom_1.pos[0]) - atom_2.pos[0], 2) +
            pow(float(atom_1.pos[1]) - atom_2.pos[1], 2) +
            pow(float(atom_1.pos[2]) - atom_2.pos[2], 2), .5)

    def GetAtoms(self):
        return self.atoms

    def GetBonds(self):
        return self.bond_list

    def __str__(self):
        return str(self.atoms) + " " + str(self.bonds) + " " + str(self.ligand)


class Atom:
    def __init__(self, symbol, pos, index, mol):
        self.symbol = symbol
        self.pos = pos
        self.index = index
        self.mol = mol
        self.aromatic = False

    def GetSymbol(self):
        return self.symbol

    def GetDegree(self):
        return len(self.mol.bonds[self.index])

    def GetTotalNumHs(self):
        return len([i for i in self.mol.bonds[self.index] if self.mol.atoms[i].symbol == "H"])

    def GetImplicitValence(self):
        return 0 # merp

    def GetIsAromatic(self):
        return self.aromatic

    def GetIdx(self):
        return self.index+1

    def __str__(self):
        return "Atom(symbol=%s, pos=%s, index=%s)" % (self.symbol, self.pos, self.index)


class Bond:
    def __init__(self, atom_1, atom_2, order, style, mol):
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.order = order
        self.style = style
        self.mol = mol

    def GetBeginAtom(self):
        return self.mol.atoms[self.atom_1]
    
    def GetEndAtom(self):
        return self.mol.atoms[self.atom_2]

    def GetBondType(self):
        if self.order == 1:
            return Chem.rdchem.BondType.SINGLE
        elif self.order == 2:
            return Chem.rdchem.BondType.DOUBLE
        elif self.order == 3:
            return Chem.rdchem.BondType.TRIPLE
        elif self.order == 1.5: 
            return Chem.rdchem.BondType.AROMATIC
        raise Exception("o shit waddap")

    def GetIsConjugated(self):
        return False # lmao

    def IsInRing(self):
        return self.order == 1.5 # currently just checking for aromacity but can make smarter later

    def __str__(self):
        return "Bond(atom1=%s, atom2=%s, order=%s, style=%s)" % (self.atom_1, self.atom_2, self.order, self.style)

def parse_crk3d_file(filename):
    molecules = []
    with open(filename, "r") as input_file:
        while True:
            try:
                molname = next(input_file).strip()
                cur_mol = Mol(molname)
            except StopIteration:
                break

            ligand_decoy = next(input_file).strip()
            if ligand_decoy == "decoy":
                cur_mol.ligand = False
            elif ligand_decoy == "ligand":
                cur_mol.ligand = True
            else:
                raise Exception("not ligand or decoy!!!!")

            cur_mol.smile = next(input_file).strip()

            assert("<Property Type=\"ModelStructure\">" == next(input_file).strip())
            assert("<Structure3D>" == next(input_file).strip())

            charge_spin = re.match(r"<Group Charge=\"([0-9]+)\" Spin=\"([0-9])+\">", next(input_file).strip())
            charge = charge_spin.group(1)
            cur_mol.charge = charge
            spin = charge_spin.group(2)
            cur_mol.spin = spin

            while True:
                first_line = next(input_file).strip()
                if re.match(r"<Atom.*", first_line):
                    atom_id = int(re.match(r"<Atom ID=\"([0-9]*)\">", first_line).group(1)) - 1
                    X = float(re.match(r"<X>([0-9\.\-]+)</X>", next(input_file).strip()).group(1))
                    Y = float(re.match(r"<Y>([0-9\.\-]+)</Y>", next(input_file).strip()).group(1))
                    Z = float(re.match(r"<Z>([0-9\.\-]+)</Z>", next(input_file).strip()).group(1))
                    element = re.match(r"<Element>([a-zA-Z]+)</Element>", next(input_file).strip()).group(1)
                    assert(next(input_file).strip() == "</Atom>")

                    cur_mol.atoms.append(Atom(element, [X, Y, Z], atom_id, cur_mol))

                elif first_line == "<Bond>":
                    atom_1 = int(re.match(r"<From>([0-9\.\-]+)</From>", next(input_file).strip()).group(1)) - 1
                    atom_2 = int(re.match(r"<To>([0-9\.\-]+)</To>", next(input_file).strip()).group(1)) - 1
                    order = float(re.match(r"<Order>([0-9\.\-]+)</Order>", next(input_file).strip()).group(1))
                    style = float(re.match(r"<Style>([0-9\.\-]+)</Style>", next(input_file).strip()).group(1))
                    assert(next(input_file).strip() == "</Bond>")

                    new_bond = Bond(atom_1, atom_2, order, style, cur_mol)
                    cur_mol.bonds[atom_1][atom_2].append(new_bond)
                    # cur_mol.bonds[atom_2][atom_1].append(new_bond)
                    cur_mol.bond_list.append(new_bond)

                elif first_line == "</Group>":
                    assert("</Structure3D>" == next(input_file).strip())
                    assert("</Property>" == next(input_file).strip())

                    break
                else:
                    raise Exception()

            molecules.append(cur_mol)

    return molecules


def compute_fingerprints_wrapper(folds, cv1_trains_tests, dud_parsed_crk3d):
    cv1_trains, cv1_tests = cv1_trains_tests
    
    for train_files, test_files, crk3d in tqdm(zip(cv1_trains, cv1_tests, dud_parsed_crk3d)):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        X, y = crk3d, [1 if molecule.ligand else 0 for molecule in crk3d]

        for (train_val_indices, test_indices), train_file, test_file in zip(skf.split(X, y), train_files, test_files):
            X_train_val, X_test = np.array(X)[train_val_indices], np.array(X)[test_indices]
            y_train_val, y_test = np.array(y)[train_val_indices], np.array(y)[test_indices]

            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, shuffle=True)

            dataset = (X_train, y_train), (X_val, y_val), (X_test, y_test)

            compute_fingerprints(dataset, train_file, test_file)

def dud_experiment():
    dud_datasets = ["ace", "ache", "alr2", "ampc", "ar"]
    dud_crk3d_files = ["dud/crk3d/%s.crk3d" % dataset for dataset in dud_datasets]
    dud_parsed_crk3d = [parse_crk3d_file(dataset) for dataset in dud_crk3d_files]

    cv1_folds=5
    fingerprint_train_all_dataset, fingerprint_test_all_dataset = [], []

    for dataset in dud_datasets:
        for fold_num in range(cv1_folds):
            make_files(folder="dud/fingerprint/"+dataset+"/"+str(fold_num))
        fingerprint_train_dataset = make_files(filenames=[("dud/fingerprint/" + dataset + "/cv_" + str(fold_num) + "/train.csv") for fold_num in range(cv1_folds)])
        fingerprint_test_dataset = make_files(filenames=[("dud/fingerprint/" + dataset + "/cv_" + str(fold_num) + "/test.csv") for fold_num in range(cv1_folds)])
        fingerprint_train_all_dataset.append(fingerprint_train_dataset)
        fingerprint_test_all_dataset.append(fingerprint_test_dataset)

    dud_result_files = make_files(folder="dud/results", filenames=[i + ".csv" for i in dud_datasets])

    # compute_fingerprints_wrapper(cv1_folds, (fingerprint_train_all_dataset, fingerprint_test_all_dataset), dud_parsed_crk3d)

    for fingerprint_train_dataset, fingerprint_test_dataset, result_filename in zip(fingerprint_train_all_dataset, fingerprint_test_all_dataset, dud_result_files):
        print("%s -> %s" % (fingerprint_train_dataset, result_filename))

        column_names = {"fingerprints": "fingerprints", "target": "target"}

        cv1_datasets = []

        for fingerprint_train, fingerprint_test in zip(fingerprint_train_dataset, fingerprint_test_dataset):
            fingerprint_train = import_data(fingerprint_train, column_names)
            fingerprint_test = import_data(fingerprint_test, column_names)
            cv1_datasets.append((fingerprint_train, fingerprint_test))

            # print(fingerprint_train[0])
            # print(fingerprint_test[0])

        # random_forest_experiment(cv1_datasets, result_filename, cv1_folds)
        svm_experiment(cv1_datasets, result_filename, cv1_folds)
        # mlp_experiment(dataset, result_filename, cv1_folds)
        # logreg_experiment(dataset, result_filename, cv1_folds)

    compile_experiment_results(dud_result_files, "f1", default_batch_num=None)

# def bagging_experiment(clf, arg, input_filename = "lxr_nobkg_fingerprints.csv"):

#     column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}
#     fingerprints, targets = import_data(input_filename, column_names)
#     with open("vecs_zinc.txt") as file:
#         file = list(file)
#         file = [line for line in file]
#         random.shuffle(file)
#         count = 0
#         for line in file[:1000]:
#             fingerprints.append(line.split(" ")[:50])
#         targets+[0]*1000
#     dataset = (fingerprints, targets)

#     f1 = bagging(dataset, 50, clf, arg)
#     plt.hist(f1, alpha = 0.5)

if __name__ == "__main__":
    # lxr_experiment()
    dud_experiment()
