from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr

from neuralfingerprint import load_data
from neuralfingerprint import build_conv_deep_net
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint.util import rmse
from sklearn.model_selection import StratifiedKFold, train_test_split

import csv

from autograd import grad

from hashlib import sha256

model_params = dict(fp_length=50,    # Usually neural fps need far fewer dimensions than morgan.
                    fp_depth=4,      # The depth of the network equals the fingerprint radius.
                    conv_width=20,   # Only the neural fps need this parameter.
                    h1_size=100,     # Size of hidden layer of network on top of fps.
                    L2_reg=np.exp(-2))
train_params = dict(num_iters=100,
                    batch_size=100,
                    init_scale=np.exp(-4),
                    step_size=np.exp(-6))

# Define the architecture of the network that sits on top of the fingerprints.
vanilla_net_params = dict(
    layer_sizes = [model_params['fp_length'], model_params['h1_size']],  # One hidden layer.
    normalize=True, L2_reg = model_params['L2_reg'], nll_func = rmse)

def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params, seed=0,
             validation_smiles=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print("Total number of weights in the network:", num_weights)
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    num_print_examples = 100
    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = []
    def callback(weights, iter):
        if iter % 10 == 0:
            train_preds = undo_norm(pred_fun(weights, train_smiles[:num_print_examples]))
            cur_loss = loss_fun(weights, train_smiles[:num_print_examples], train_targets[:num_print_examples])
            training_curve.append(cur_loss)
            print("Iteration", iter, "loss", cur_loss,\
                "train RMSE", rmse(train_preds, train_raw_targets[:num_print_examples]),\
                "max of weights", np.max(np.abs(weights)), end="")
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print("Validation RMSE", iter, ":", rmse(validation_preds, validation_raw_targets), end="")
            print("")

    # Build gradient using autograd.
    print("gradding")
    grad_fun = grad(loss_fun)

    grad_fun_with_data = build_batched_grad(grad=grad_fun, batch_size=train_params['batch_size'], inputs=train_smiles, targets=train_targets)
    # grad_fun_with_data is function that computes the gradient of the loss given the weights and the batch number (to determine training weights and targets)

    print("optimizing")
    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback, num_iters=train_params['num_iters'], step_size=train_params['step_size'])

    print("optimized")

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve

def compute_fingerprints(dataset, train_file, test_file, learning_rate):
    train, val, test = dataset
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    X_train_val = np.concatenate((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    
    global train_params
    # train_params["num_iters"] = int(len(X_train)/train_params["batch_size"])
    train_params["step_size"] = learning_rate

    smiles_to_fps = {}
    conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
    conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                        'fp_length' : model_params['fp_length'],
                        'normalize' : 1,
                        'smiles_to_fps': smiles_to_fps}

    loss_fun, pred_fun, conv_parser = build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
    num_weights = len(conv_parser)

    predict_func, trained_weights, conv_training_curve = train_nn(pred_fun, loss_fun, num_weights, X_train, y_train, train_params, validation_smiles=X_val, validation_raw_targets=y_val)

    pred_fun(trained_weights, X_train_val)

    with open(train_file, "w+") as smiles_fps_file:
        header = ["smiles", "fingerprints", "target"]
        file_info = [[smile, smiles_to_fps[smile], target] for smile, target in zip(X_train_val, y_train_val)]

        writer = csv.writer(smiles_fps_file)
        writer.writerow(header)
        for line in file_info:
            writer.writerow(line)

    predict_func(X_test)
    with open(test_file, "w+") as smiles_fps_file:
        header = ["smiles", "fingerprints", "target"]
        file_info = [[smile, smiles_to_fps[smile], target] for smile, target in zip(X_test, y_test)]

        writer = csv.writer(smiles_fps_file)
        writer.writerow(header)
        for line in file_info:
            writer.writerow(line)
