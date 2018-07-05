Neural Graph Fingerprints
=============

<img src="https://github.com/HIPS/DeepMolecules/blob/master/paper/figures/3d-nets/net1.png" width="300">

 For pacakge installation help, https://github.com/HIPS/neural-fingerprint

The original code for generating fingerprints is in neural-fingerprint/neuralfingerprint/build_convnet.py. On line 96 

return sum(all_layer_fps), atom_activations, array_rep

and sum(all_layer_fps) is the fingerprint we are after.

The simplified code is compute_fingerprint_and_conv_baseline.py. Currently, the input file is lxr_nobkg.csv, and if you want to input other files, make sure to change N_train,N_val & N_test correspondingly.


