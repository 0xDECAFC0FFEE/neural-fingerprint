import csv
import random
import ast
from collections import defaultdict
import time
import copy
import os

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

def log_experiment(results, filename, default_header=[], overwrite=False, copy_raw=False):
    if overwrite:
        open(filename, "w+").close()
    else:
        open(filename, "a+").close()
    with open(filename, "r") as log_file:
        try:
            csv_reader = csv.reader(log_file)
            header = next(csv_reader)
            log_file.seek(0)
            data = csv.DictReader(log_file)
            data = [defaultdict(lambda: "", line) for line in data]
        except:
            header = []
            data = []

    results = [defaultdict(lambda: "", line) for line in results]

    if not copy_raw:
        for line in results:
            line["timestamp"] = time.strftime("%I:%M %m-%d")
            try:
                line["batch_num"] = log_experiment.batch_number
            except:
                log_experiment.batch_number = random.randint(0, 99999999)
                line["batch_num"] = log_experiment.batch_number

    keys_in_results = (list(results[0].keys()) + default_header) if results else list(default_header)

    header = sorted(set(list(header) + keys_in_results))
    data = data + results

    with open(filename, "w") as log_file:
        csv_writer = csv.DictWriter(log_file, header, "")
        csv_writer.writeheader()
        for line in data:
            csv_writer.writerow(line)

def average_scores(scores, target=None):
    scores_to_include = {#"accuracy", "TP", "TN", "FP", "FN", "log_loss", "rocNOT50", "bedroc",
        "TPR", "TNR", "f1", "roc", "batch_num"}
    average_score = defaultdict(lambda: float(0))
    for score in scores:
        for key, value in score.items():
            if key not in scores_to_include:
                continue
            average_score[key] += ast.literal_eval(value)
    for key, value in average_score.items():
        average_score[key] /= len(scores)
    return dict(average_score)

def compile_experiment_results(input_files, target, default_batch_num=None):
    """
    compiles results of experiments by classifier type
    crawls through each dataset, finds the batch number of its most recent run.
    filters for most recent batch and finds best result for each classifier
    """
    all_results = defaultdict(lambda: [])
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

            for clf in batch_cls:
                all_results[(clf, filename)] = [i for i in cur_batch if i["classifier"] == clf]

            for result in average_expr_results:
                result["filename"] = filename

            results.extend(copy.deepcopy(average_expr_results))

    default_header = sorted(default_header)

    print(len(batch_nums))

    if len(batch_nums) == 1 or default_batch_num:
        batch_num = batch_nums.pop() if not default_batch_num else default_batch_num
        output_file = "compiled_experiment_%s_results.csv" % batch_num
        print("batch id: ", batch_num)
    else:
        output_file = "compiled_experiment_results.csv"
        print("batch ids: ", batch_nums)

    log_experiment(results, output_file, default_header=list(default_header), overwrite=True, copy_raw=True)

    scores = {}
    for clf in ["RandomForestClassifier", "SVC", "MLPClassifier", "LogisticRegression"]:
        datasets = ["ace", "ache", "alr2", "ampc", "ar"]
        
        clf_results = defaultdict(lambda: [])
        for dataset, dataset_file in [(dataset, "dud/results/%s.csv"%dataset) for dataset in datasets]:
            clf_results[dataset].extend([float(i["roc"]) for i in all_results[(clf, dataset_file)]])

        scores[clf] = dict(clf_results)

    # os.system("open %s" % output_file)
    plot(scores, batch_num)


def plot(scores, exp_id):
    print(scores)
    import matplotlib
    import matplotlib.pyplot as plt
    import itertools

    # matplotlib.rc("font", size=20)

    xs, ys = defaultdict(lambda:[]), defaultdict(lambda:[])
    avg_xs, avg_ys = defaultdict(lambda:[]), defaultdict(lambda:[])
    x_axis_labels = []

    for clf_name, clf_scores in scores.items():
        for dataset, data in clf_scores.items():
            xs[clf_name].extend(["%s-%s" % (clf_name, dataset)]*len(data))
            ys[clf_name].extend(data)

            avg_xs[clf_name].append("%s-%s" % (clf_name, dataset))
            avg_ys[clf_name].append(sum(data)/float(len(data)))

            x_axis_labels.extend([dataset])

    plt.clf()

    for clf in xs:
        plt.scatter(xs[clf], ys[clf])

    for clf in avg_xs:
        plt.scatter(avg_xs[clf], avg_ys[clf], color="black", marker="x")

    plt.legend(list(xs)+["average roc score"])

    plt.ylabel('ROC score')

    plt.xticks(range(20), x_axis_labels, rotation=-40)

    title = 'experiment %s' % exp_id
    plt.title(title)

    plt.rcParams["figure.figsize"] = (20,3)

    with open(title+".png", "w+b") as file:
        plt.savefig(file, dpi=200)

    plt.show()
    
    # os.system("open '%s.png'" % title)
