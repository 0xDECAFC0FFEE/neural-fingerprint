import autograd.numpy as np
import autograd.numpy.random as npr

from neuralfingerprint import load_data
from neuralfingerprint import build_morgan_deep_net
from neuralfingerprint import build_conv_deep_net
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint.util import rmse

import csv

from autograd import grad

# task_params = {'target_name' : 'measured log solubility in mols per litre',
#                'data_file'   : 'examples/delaney.csv'}
# N_train = 800
# N_val = 20
# N_test = 20

task_params = {'target_name': 'LXRbeta binder',
               'data_file': 'dud/ace_actives.smi'}
N_train = 131
N_val   = 10
N_test  = 0

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
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve


def main(task_params, train_val_test_split, output_filename):
    N_train, N_val, N_test = train_val_test_split
    print "Loading data..."
    traindata, valdata, testdata = load_data(
        task_params['data_file'], (N_train, N_val, N_test),
        input_name='smiles', target_name=task_params['target_name'])
    train_inputs, train_targets = traindata
    val_inputs,   val_targets   = valdata
    test_inputs,  test_targets  = testdata

    def run_conv_experiment():
        smiles_to_fps = {}

        conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
        conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1,
                            'smiles_to_fps': smiles_to_fps}
        loss_fun, pred_fun, conv_parser = \
            build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
        num_weights = len(conv_parser)
        predict_func, trained_weights, conv_training_curve = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                     train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)

        with open(output_filename, "w+") as smiles_fps_file:
            writer = csv.writer(smiles_fps_file, lineterminator='\n')

            all_inputs = list(train_inputs) + list(val_inputs) + list(test_inputs)
            all_targets = list(train_targets) + list(val_targets) + list(test_targets)

            file = [["smiles", "fingerprints", task_params['target_name']]]
            for smile, target in sorted(zip(all_inputs, all_targets)):
                try:
                    file.append([smile, smiles_to_fps[smile], target])
                except:
                    print("skipping smile %s in fingerprint computation" % smile)
            
            for line in file:
                writer.writerow(line)

    run_conv_experiment()

if __name__ == '__main__':
    smi_to_csv("dud/ace_actives.smi", "dud/ace_background.smi")
    main(task_params, N_train, N_val, N_test)
