from sklearn.metrics import log_loss, matthews_corrcoef, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

def compute_validation_weights(test_y):
    num_pos_val = sum(test_y)
    num_neg_val = len(test_y) - num_pos_val
    if num_pos_val > 0:
        pos_weight, neg_weight = float(num_neg_val) / float(num_pos_val), 1
    else:
        pos_weight, neg_weight = 1, 1
    validation_weights = [pos_weight if i == 1 else neg_weight for i in test_y]
    return validation_weights

def interpret_score(pred_y_proba, test_y, validation_weights=None, show_roc=False):
    pred_y_pos = [probs[1] / (probs[0] + probs[1]) for probs in pred_y_proba]
    pred_y_class = [prob_pos > .5 for prob_pos in pred_y_pos]

    if validation_weights == None:
        validation_weights = compute_validation_weights(test_y)

    TP = sum([1 for test, pred in zip(test_y, pred_y_class) if test == pred and pred == 1])
    FP = sum([1 for test, pred in zip(test_y, pred_y_class) if test != pred and pred == 1])
    TN = sum([1 for test, pred in zip(test_y, pred_y_class) if test == pred and pred == 0])
    FN = sum([1 for test, pred in zip(test_y, pred_y_class) if test != pred and pred == 0])
    accuracy = float(TP+TN) / float(TP+FP+TN+FN)

    f1 = f1_score(test_y, pred_y_class, sample_weight=validation_weights)

    roc = roc_auc_score(test_y, pred_y_pos)
    # print(sorted(list(zip(pred_y_pos, test_y))))

    result =  {
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TPR": TP/float(TP+FN),
        "TNR": TN/float(TN+FP),
        "f1": f1,
        "roc": roc,
    }

    return result

def max_scores(scores, target):
    max_score = 0
    for score in scores:
        if score[target] > max_score:
            max_score = score
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
