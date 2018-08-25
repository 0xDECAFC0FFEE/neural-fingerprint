import autograd.numpy as np
import autograd.numpy.random as npr

from neuralfingerprint import load_data
from neuralfingerprint import build_morgan_deep_net
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
    print "Total number of weights in the network:", num_weights
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    num_print_examples = 100
    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = []
    def callback(weights, iter):
        if iter % 10 == 0:
            print "max of weights", np.max(np.abs(weights))
            train_preds = undo_norm(pred_fun(weights, train_smiles[:num_print_examples]))
            cur_loss = loss_fun(weights, train_smiles[:num_print_examples], train_targets[:num_print_examples])
            training_curve.append(cur_loss)
            print "Iteration", iter, "loss", cur_loss,\
                  "train RMSE", rmse(train_preds, train_raw_targets[:num_print_examples]),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", iter, ":", rmse(validation_preds, validation_raw_targets),

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    print("adaming")
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'])
    print("adamed")

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve

def compute_fingerprints(dataset, train_file, test_file):
    train, val, test = dataset
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    smiles_to_fps = {}
    conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
    conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                        'fp_length' : model_params['fp_length'], 'normalize' : 1,
                        'smiles_to_fps': smiles_to_fps}
    loss_fun, pred_fun, conv_parser = \
        build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
    num_weights = len(conv_parser)
    predict_func, trained_weights, conv_training_curve = \
        train_nn(pred_fun, loss_fun, num_weights, X_train, y_train,
                train_params, validation_smiles=X_val, validation_raw_targets=y_val)

    X_train_val = np.concatenate(X_train, X_val)
    y_train_val = np.concatenate(y_train, y_val)

    pred_fun(trained_weights, X_train_val)

    with open(train_file, "w+") as smiles_fps_file:
        header = ["smiles", "fingerprints", "target"]
        file_info = [[smile.smile, smiles_to_fps[smile], target] for smile, target in zip(X_train_val, y_train_val)]

        writer = csv.writer(smiles_fps_file)
        writer.writerow(header)
        for line in file_info:
            writer.writerow(line)

    predict_func(X_test)

    with open(test_file, "w+") as smiles_fps_file:
        header = ["smiles", "fingerprints", "target"]
        file_info = [[smile.smile, smiles_to_fps[smile], target] for smile, target in zip(X_test, y_test)]

        writer = csv.writer(smiles_fps_file)
        writer.writerow(header)
        for line in file_info:
            writer.writerow(line)

def distance_multiplier(mol, atom_1_index, atom_2_index):
    return 1.0/float(mol.distance(atom_1_index, atom_2_index))

def compute_neighbor_multipliers(mol):
    distance_mapping = {}
    for atom_1 in len(mol.atoms):
        distance_mapping[atom_1] = {}

        for atom_2 in len(mol.atoms):
            if atom_1 == atom_2:
                continue
            if mol.distance(atom_1, atom_2) > 3:
                continue
            
            distance_mapping[atom_1][atom_2] = distance_multiplier(mol, atom_1, atom_2)
    
    return distance_mapping

            
def atom_symbol_to_num(symbol):
    # expecting symbol like "O", "CL", "he", or "Fr". 

    normalized_symbol = symbol.strip().title()
    
    return float(int(sha256(normalized_symbol.encode('utf-8')).hexdigest(), 16) % 100000)

def interpolate_fingerprints(train_val_test_split, output_filename, example_file, example_target_column, target_filename):
    print "Loading data..."
    traindata, valdata, _ = load_data(
        example_file, train_val_test_split, input_name='smiles', target_name=example_target_column)
    train_inputs, train_targets = traindata
    val_inputs,   val_targets = valdata

    target_inputs = []
    with open(target_filename) as target_file:
        reader = csv.DictReader(target_file)
        for line in reader:
            target_inputs.append(line["smiles"])

    smiles_to_fps = {}
    conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
    conv_arch_params = {'num_hidden_features': conv_layer_sizes,
                        'fp_length': model_params['fp_length'], 'normalize': 1,
                        'smiles_to_fps': smiles_to_fps}
    loss_fun, pred_fun, conv_parser = build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
    num_weights = len(conv_parser)
    predict_func, trained_weights, conv_training_curve = \
        train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                 train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)

    pred_fun(trained_weights, target_inputs)

    with open(output_filename, "w+") as smiles_fps_file:
        header = ["smiles", "fingerprints"]
        file_info = [[smile, smiles_to_fps[smile]] for smile in sorted(target_inputs)]

        writer = csv.writer(smiles_fps_file)
        writer.writerow(header)
        for line in file_info:
            writer.writerow(line)

if __name__ == '__main__':
    # compute_fingerprints(
    #     train_val_test_split=(95, 45, 0),
    #     data_file='lxr_nobkg.csv',
    #     data_target_column='LXRbeta binder',
    #     output_filename='lxr_nobkg_fingerprints.csv'
    # )
    interpolate_fingerprints(
        train_val_test_split=(95, 45, 0),
        output_filename='top1000_rf_fingerprints.csv',
        example_file='lxr_nobkg.csv',
        example_target_column='LXRbeta binder',
        target_filename='top1000_rf_smiles.csv'
    )
