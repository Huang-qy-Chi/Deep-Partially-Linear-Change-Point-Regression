import numpy as np
import time
from seed import set_seed
from data_generator import generate_case_Deep
from one_case_deep import _one_case_deep
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # setting
    Theta = [-1,2]; eta = 2; corr = 0.5
    # other parameter for iteration
    maxloop = 100; seq = 0.01; m0 = 2
    # seed, number of replication, n_jobs = max(1, cpu_count()-nj)
    Seed = 1145; B = 20; nj = 1
    n = [300, 500, 1000, 2000]

    # hyperparameter, require further adjust
    n_layer = [3,3,3,3]; n_layer_1 = [3,3,3,3]; n_layer_2 =[3,3,3,3]
    n_node = [50,50,50,50]; n_node_1 = [50,50,50,50]; n_node_2 = [50,50,50,50]
    n_lr = [1e-3,1e-3,1e-3,1e-3]; n_lr_1 = [2e-3,2e-3,2e-3,2e-3]; n_lr_2 = [2e-3,2e-3,2e-3,2e-3]
    n_epoch = [200,200,200,200]; n_epoch_1 = [200,200,200,200]; n_epoch_2 = [200,200,200,200]
    patiences = [20,20,20,20]; patiences_1 = [20,20,20,20]; patiences_2 = [20,20,20,20]

    # test data
    set_seed(14)
    test_data = generate_case_Deep(500,corr,Theta,eta)
    X_test = test_data['X']
    Z_test = test_data['Z']
    A_test = test_data['A']
    Y_test = test_data['Y']
    f_true = test_data['f_X']
    g_true = test_data['g_X']
    Res_true = test_data['f_X_C']


    for i in range(len(n)):
        print(f"{"Start"}_{"n"}_{n[i]}_{"case"}_{"Deep"}")
        res = _one_case_deep(Seed, n[i], corr, Theta, eta, test_data, \
                    n_layer[i], n_node[i], n_lr[i], n_epoch[i], patiences[i], \
                            n_layer_1[i], n_node_1[i], n_lr_1[i], n_epoch_1[i], patiences_1[i],\
                            n_layer_2[i], n_node_2[i], n_lr_2[i], n_epoch_2[i], patiences_2[i], \
                                    maxloop, seq, m0, B, nj, save = True)    
        print(f"{"Complete"}_{"n"}_{n[i]}_{"case"}_{"Deep"}")
        print("System pause!")
        time.sleep(1)
        print('System continue!')


    print('Mission accomplish!')





