import os
from multiprocessing import get_context, cpu_count, Manager
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
from one_run_deep import _one_run_deep



#%%-----------------one case deep------------------------
def _one_case_deep(Seed, n, corr, Theta, eta, test_data, \
                n_layer, n_node, n_lr, n_epoch, patiences, \
                        n_layer_1, n_node_1, n_lr_1, n_epoch_1, patiences_1,\
                        n_layer_2, n_node_2, n_lr_2, n_epoch_2, patiences_2, \
                                maxloop=100, seq=0.01, m0=2, B = 200, nj = 2, save = False):
    start_time = time.time()
    n_jobs = max(1, cpu_count() - nj)
    ctx = get_context("spawn")  # 跨平台更稳
    with ctx.Pool(processes=n_jobs) as pool:
        args_iter = [(i, Seed, n, corr, Theta, eta, test_data, \
                n_layer, n_node, n_lr, n_epoch, patiences, \
                        n_layer_1, n_node_1, n_lr_1, n_epoch_1, patiences_1,\
                        n_layer_2, n_node_2, n_lr_2, n_epoch_2, patiences_2, \
                                maxloop, seq, m0) for i in range(B)]
        results = pool.starmap(_one_run_deep, args_iter)

    # 拆回原来的 list
    Theta_d, eta_d, re_f_d, re_g_d, se1_d, se2_d, \
        Theta_l, eta_l, re_f_l, re_g_l, se1_l, se2_l = map(list, zip(*results))

    # Deep change point
    # beta
    Theta_b_d = np.array(Theta_d)[:,0]
    bias_b_d = (np.mean(np.array(Theta_b_d))-Theta[0])
    sse_b_d = (np.sqrt(np.mean((np.array(Theta_b_d)-np.mean(np.array(Theta_b_d)))**2)))
    ese_b_d = (np.mean(np.array(se1_d)))
    cp_b_d = (np.mean((np.array(Theta_b_d)-1.96*np.array(se1_d)<=Theta[0])*\
                       (Theta[0]<=np.array(Theta_b_d)+1.96*np.array(se1_d))))
    # gamma
    Theta_g_d = np.array(Theta_d)[:,1]
    bias_g_d = (np.mean(np.array(Theta_g_d))-Theta[1])
    sse_g_d = (np.sqrt(np.mean((np.array(Theta_g_d)-np.mean(np.array(Theta_g_d)))**2)))
    ese_g_d = (np.mean(np.array(se2_d)))
    cp_g_d = (np.mean((np.array(Theta_g_d)-1.96*np.array(se2_d)<=Theta[1])*\
                       (Theta[1]<=np.array(Theta_g_d)+1.96*np.array(se2_d))))
    
    # eta
    bias_eta_d = (np.mean(np.array(eta_d))-eta)
    sse_eta_d = (np.sqrt(np.mean((np.array(eta_d)-np.mean(np.array(eta_d)))**2)))

    # Re
    # f
    RE_f_d = np.mean(np.array(re_f_d))
    SD_ref_d =  (np.sqrt(np.mean((np.array(re_f_d)-np.mean(np.array(re_f_d)))**2)))

    # g
    RE_g_d = np.mean(np.array(re_g_d))
    SD_reg_d =  (np.sqrt(np.mean((np.array(re_g_d)-np.mean(np.array(re_g_d)))**2)))

    # Linear change point
    # beta
    Theta_b_l = np.array(Theta_l)[:,0]
    bias_b_l = (np.mean(np.array(Theta_b_l))-Theta[0])
    sse_b_l = (np.sqrt(np.mean((np.array(Theta_b_l)-np.mean(np.array(Theta_b_l)))**2)))
    ese_b_l = (np.mean(np.array(se1_l)))
    cp_b_l = (np.mean((np.array(Theta_b_l)-1.96*np.array(se1_l)<=Theta[0])*\
                       (Theta[0]<=np.array(Theta_b_l)+1.96*np.array(se1_l))))
    # gamma
    Theta_g_l = np.array(Theta_l)[:,1]
    bias_g_l = (np.mean(np.array(Theta_g_l))-Theta[1])
    sse_g_l = (np.sqrt(np.mean((np.array(Theta_g_l)-np.mean(np.array(Theta_g_l)))**2)))
    ese_g_l = (np.mean(np.array(se2_l)))
    cp_g_l = (np.mean((np.array(Theta_g_l)-1.96*np.array(se2_l)<=Theta[1])*\
                       (Theta[1]<=np.array(Theta_g_l)+1.96*np.array(se2_l))))
    
    # eta
    bias_eta_l = (np.mean(np.array(eta_l))-eta)
    sse_eta_l = (np.sqrt(np.mean((np.array(eta_l)-np.mean(np.array(eta_l)))**2)))

    # Re
    # f
    RE_f_l = np.mean(np.array(re_f_l))
    SD_ref_l =  (np.sqrt(np.mean((np.array(re_f_l)-np.mean(np.array(re_f_l)))**2)))

    # g
    RE_g_l = np.mean(np.array(re_g_l))
    SD_reg_l =  (np.sqrt(np.mean((np.array(re_g_l)-np.mean(np.array(re_g_l)))**2)))

    if save==True:
        result_linear5 = {
            'Theta': Theta_l,
            'eta': eta_l,
            'Re_f': re_f_l,
            'Re_g': re_g_l,
            'se1': se1_l,
            'se2': se2_l, 
        }
        filename = f"{"ZCPDPLM"}_{5}_{"n"}_{n}_{"deep"}.npy"
        np.save(filename, result_linear5, allow_pickle=True)
            
        result_deep5 = {
            'Theta': Theta_d,
            'eta': eta_d,
            'Re_f': re_f_d,
            'Re_g': re_g_d,
            'se1': se1_d,
            'se2': se2_d, 
        }
        filename = f"{"ZCPLM"}_{5}_{"n"}_{n}_{"deep"}.npy"
        np.save(filename, result_deep5, allow_pickle=True)

    result_index = {
        'bias_b_d': bias_b_d,
        'sse_b_d': sse_b_d,
        'ese_b_d': ese_b_d,
        'cp_b_d': cp_b_d, 
        'bias_g_d': bias_g_d,
        'sse_g_d': sse_g_d,
        'ese_g_d': ese_g_d,
        'cp_g_d': cp_g_d, 
        'RE_f_d': RE_f_d,
        'RE_g_d': RE_g_d,
        'SD_f_d': SD_ref_d,
        'SD_g_d': SD_reg_d,
        'bias_eta_d': bias_eta_d,  
        'sse_eta_d':sse_eta_d, #case deep
        'bias_b_l': bias_b_l,
        'sse_b_l': sse_b_l,
        'ese_b_l': ese_b_l,
        'cp_b_l': cp_b_l, 
        'bias_g_l': bias_g_l,
        'sse_g_l': sse_g_l,
        'ese_g_l': ese_g_l,
        'cp_g_l': cp_g_l,
        'RE_f_l': RE_f_l,
        'RE_g_l': RE_g_l,
        'SD_f_l': SD_ref_l,
        'SD_g_l': SD_reg_l,
        'bias_eta_l': bias_eta_l,
        'sse_eta_l':sse_eta_l #case deep
    }

    if save==True:
        filename = f"{"AResult_index"}_{"n"}_{n}_{"deep"}.npy"
        np.save(filename, result_index, allow_pickle=True)

    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds for estimation.")
    return result_index



# An example, for the main function Main_deep.py
if __name__ == '__main__':
    from seed import set_seed
    from data_generator import generate_case_Deep
    Theta = [-1,2]; eta = 2; corr = 0.5
    n = 1000; maxloop = 100; seq = 0.01; m0 = 2
    Seed = 1145; B = 20; nj = 2

    # hyperparameter
    n_layer = 3; n_layer_1 = 3; n_layer_2 = 3
    n_node = 50; n_node_1 = 50; n_node_2 = 50
    n_lr = 1e-3; n_lr_1 = 2e-3; n_lr_2 = 2e-3
    n_epoch = 200; n_epoch_1 = 200; n_epoch_2 = 200
    patiences = 20; patiences_1 = 20; patiences_2 = 20

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

    # test: 
    res = _one_case_deep(Seed, n, corr, Theta, eta, test_data, \
                n_layer, n_node, n_lr, n_epoch, patiences, \
                        n_layer_1, n_node_1, n_lr_1, n_epoch_1, patiences_1,\
                        n_layer_2, n_node_2, n_lr_2, n_epoch_2, patiences_2, \
                                maxloop, seq, m0, B, nj, save = True)
    print(res)
