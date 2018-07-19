import csv
import ast
from collections import namedtuple, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, matthews_corrcoef
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

def random_fetch(filename):

    with open(filename) as file:
        file = list(file)
        file = [line for line in file]
        random.shuffle(file)
        count = 0
        for line in file:
            yield line
            count += 1
            if count >=1000:
                print "start shuffling"
                random.shuffle(file)
                print "finish shuffling"
                count = 0

corr = []
gen = random_fetch("vecs_zinc.txt")
roc = []
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

def interpret_score(pred_y_proba, test_y, validation_weights=None, pred_y=None):
    if pred_y == None:
        pred_y = [max([(index, i) for i, index in enumerate(probs)])[1] for probs in pred_y_proba]

# weighted log loss function(aka cross entropy loss) pos weigh more than neg
# f score
# skip mcc

    TP = sum([1 for test, pred in zip(test_y, pred_y) if test == pred and pred == 1])
    FP = sum([1 for test, pred in zip(test_y, pred_y) if test != pred and pred == 1])
    TN = sum([1 for test, pred in zip(test_y, pred_y) if test == pred and pred == 0])
    FN = sum([1 for test, pred in zip(test_y, pred_y) if test != pred and pred == 0])
    accuracy = float(TP+TN) / float(TP+FP+TN+FN)
    mcc = 0 # matthews_corrcoef(test_y, pred_y)
    
    if validation_weights != None:
        weighted_log_loss = log_loss(test_y, pred_y_proba, sample_weight=validation_weights)
    else:
        weighted_log_loss = log_loss(test_y, pred_y_proba)
    # f_score = 
    return {
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "MCC": mcc,
        "log_loss": weighted_log_loss
    }

def roc_50(pred_y, true_y):
	y = zip(pred_y,true_y)
	p,n = 0,0
	for target in true_y:
		if target == 1:
			p +=1.0
		else:
			n +=1.0
	y.sort(key = lambda f: f[0], reverse = True)
	tp,fp = 0,0
	fpr,tpr = [0.0],[0.0]
	for pred,target in y:
		if target == 1:
			tp +=1.0
			tpr.append(tp/p)
			fpr.append(fp/n)
		else:
			fp +=1.0
			tpr.append(tp/p)
			fpr.append(fp/n)
		if fp >= 50:
			break
	score =metrics.auc(fpr,tpr)/fpr[-1]
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

def arb_sampling(filename, clf, dataset):
    pos_train_X = []
    pos_train_y = []
    neg_train_X = []
    neg_train_y = []
    #gen = random_fetch(filename)
    
    for fingerprint, target in zip(dataset[0], dataset[1]):
        if target == 1:
            pos_train_X.append(fingerprint)
            pos_train_y.append(target)
        else:
            neg_train_X.append(fingerprint)
            neg_train_y.append(target)

    pos = len(pos_train_X)
    neg = len(neg_train_X)
   
    neg_train_X = neg_train_X[:pos/3]
    neg_train_y = neg_train_y[:pos/3]
    neg = len(neg_train_X)
    
    X = [[next(gen).split(" ")[:50] for j in range(pos - neg)] for i in range(10)]
    for i in range(10):
       
        train = np.array(pos_train_X + X[i] + neg_train_X).astype(float)
        target = np.array(pos_train_y + [0]*(pos - neg) + neg_train_y).astype(float)
        
        clf[i].fit(train, target)

    return clf 
    
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

def fit_score_classifier(arguments, classifier_type, dataset, validation_weights):
    ((train_X, train_y), (test_X, test_y)) = dataset
    
    classifier = classifier_type(**arguments)
    classifier.fit(train_X, train_y)
    pred_y = classifier.predict_proba(test_X)
    score = interpret_score(pred_y, test_y, validation_weights=validation_weights)
    
    return score

def cv_layer_2(arguments, classifier_type, dataset, folds):
    fingerprints, targets = dataset
    '''
    random.seed(datetime.now())
    train_X, test_X, train_y, test_y = train_test_split(fingerprints, targets, \
    								test_size = 0.2, random_state = random.randint(0, 9999999))
    '''
    skf = StratifiedKFold(n_splits = folds, shuffle = True)
   
    max_score, max_arg = None, None
    for train_index, test_index in skf.split(fingerprints, targets):
        train_X, test_X = np.array(fingerprints)[train_index], np.array(fingerprints)[test_index]
        train_y, test_y = np.array(targets)[train_index], np.array(targets)[test_index]
        
        for argument in arguments:
                random.seed(datetime.now())
                argument["random_state"] = random.randint(0, 9999999)
                
                train_val_dataset = ((train_X, train_y), (test_X, test_y))
                score = sampling(argument, classifier_type, train_val_dataset)

                if max_score == None:
                    max_score, max_arg = (score["accuracy"], argument)
                elif "n_estimators" in max_arg and "max_depth" in max_arg:
                    if max_arg["n_estimators"] + max_arg["max_depth"] > argument["n_estimators"] + argument["max_depth"]:
                        max_score, max_arg = (score, argument)
                elif max_score <= score["accuracy"]:
                    max_score, max_arg = (score["accuracy"], argument)
    #print max_score                          
    return max_arg


def cv_layer_1(arguments, classifier_type, dataset, folds):
    fingerprints, targets = dataset
    
    '''
    random.seed(datetime.now())
    train_X, test_X, train_y, test_y = train_test_split(fingerprints, targets, \
									test_size = 0.2,  random_state = random.randint(0, 9999999))
    '''
    roc50 = []
    skf = StratifiedKFold(n_splits = folds, shuffle = True)
    argument_scores = []
    for train_index, test_index in skf.split(fingerprints, targets):
        train_X, test_X = np.array(fingerprints)[train_index], np.array(fingerprints)[test_index]
        train_y, test_y = np.array(targets)[train_index], np.array(targets)[test_index]
            
        dataset_layer_2 = (train_X, train_y)
        
        best_arg = cv_layer_2(arguments, classifier_type, dataset_layer_2, folds)
        '''
        clfs = [classifier_type(**best_arg) for i in range(10)]
        clfs = arb_sampling("vecs_zinc.txt", clfs, (train_X, train_y))
        pred_y = np.array([clf.predict_proba(test_X) for clf in clfs]).mean(axis = 0).tolist()
        num_pos_val = sum(train_y)
        num_neg_val = len(train_y) - num_pos_val
        pos_weight, neg_weight = float(num_neg_val)/float(num_pos_val),1
        sample_weight = [pos_weight if val == 1 else neg_weight for val in train_y]
        '''
        clf = BaggingClassifier(classifier_type(**best_arg))
        clf.fit(train_X, train_y)
        
        pred_y = clf.predict_proba(test_X)
        pred_pos = [prob[1] for prob in pred_y]
        
        roc50.append(roc_50(pred_pos, test_y))

        '''
        corr_test = np.array([next(gen).split(" ")[:50] for j in range(1000)],dtype = float)
        corr_pred = np.array([clf.predict_proba(corr_test) for clf in clfs]).mean(axis = 0).tolist()
        corr.append([i[0] for i in corr_pred])
        '''
        score = interpret_score(pred_y, test_y)
        score["roc50"] = roc50[-1]
        print roc50[-1]
        argument_scores.append(copy.deepcopy((best_arg, score)))
        
    roc.append(sum(roc50)/folds)
    
    print roc[-1]
    
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
        
    score_arguments, clf = cv_layer_1(arguments, classifier_type, dataset, folds)
    print("score_arguments %s" % score_arguments)

    results = []
    for score, argument in score_arguments:
        #print(score)
        #print(argument)
        result = dict(**score)
        result.update(**argument)
        result["classifier"] = classifier_type.__name__

        #print(result)
        results.append(result)
    
    #print(results)
    log_experiment(results, output_log)

    return score_arguments, clf


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
	
def mlp_experiment(dataset, output_log):
    classifier = MLPClassifier
    classifier_inputs_list = [{
                "solver": ['lbfgs'],
        "hidden_layer_sizes": [10],
        "alpha": [.0001,.001,.01,.1,10,100]
    }]
    folds = 5
    results = []
    clfs = []
    for classifier_inputs in classifier_inputs_list:
        output, clf = experiment(dataset, classifier, classifier_inputs, folds, output_log)
        results.extend(output)
        clfs.append(clf)
    return results, clfs
    
def logreg_experiment(dataset, output_log):
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
        output, clf = experiment(dataset, classifier, classifier_inputs, folds, output_log)
        results.extend(output)
        clfs.append(clf)
    return results, clfs
    
def lxr_experiment():
    input_filename = "lxr_nobkg_fingerprints.csv"
    output_filename = "adaboost.csv"
    column_names = {"fingerprints": "fingerprints", "target": "LXRbeta binder"}

    dataset = import_data(input_filename, column_names)

    # random_forest_experiment(dataset, output_filename)
    # svm_experiment(dataset, output_filename)
    results, clfs = mlp_experiment(dataset, output_filename)
    # results, clfs = logreg_experiment(dataset, output_filename)
    return clfs
    
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
    #dud_datasets = ["ace"]
    dud_raw_files = [("dud/%s_actives.smi" % dataset, "dud/%s_background.smi" % dataset)
        for dataset in dud_datasets]
    dud_smile_csv_files = ["dud/smiles/%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/smiles")
    make_files(dud_smile_csv_files)
    dud_fingerprint_files = ["dud/fingerprints/%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/fingerprints")
    make_files(dud_fingerprint_files)
    dud_result_files = ["dud/results/RegvsNN_%s.csv"%dataset for dataset in dud_datasets]
    make_folder("dud/results")
    make_files(dud_result_files)

    print("running fingerprint experiments")
    for fingerprint_filename, result_filename in zip(dud_fingerprint_files, dud_result_files):
        print("%s -> %s" % (fingerprint_filename, result_filename))

        column_names = {"fingerprints": "fingerprints", "target": "target"}

        fingerprints, targets = import_data(fingerprint_filename, column_names)
        
        dataset = zip(fingerprints, targets)
        random.shuffle(dataset)
        sample_data = []
        pos = 0
        neg = 0
        for fingerprint, target in dataset:
            if target==1:
                sample_data.append((fingerprint,1))
                pos +=1
            elif neg<750:
                sample_data.append((fingerprint,0))
                neg+=1
        print pos, neg     
        sample_data = zip(*sample_data)      
        
        # random_forest_experiment(dataset, result_filename)
        # svm_experiment(dataset, result_filename)
        # mlp_experiment(sample_data, result_filename)
       
        logreg_experiment(sample_data, result_filename)
        print zip(dud_datasets, roc)
    
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
    #plt.show()
    #lxr_experiment()
    #plt.plot(corr[0],corr[1], 'ro')
    #print pearsonr(corr[0], corr[1])
    #plt.show()
    dud_experiment()

