import os
from multiprocessing import get_context, cpu_count, Manager
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
from seed import set_seed
from data_generator import generate_case_Deep
from cpdplm_estimation import CPDPLM
from cpalm_estimation import CPALM
from cplm_estimation import CPLM
from LFDCP import LFDCP


#%%----------------------One run deep------------------------------
def _one_run_deep(i, Seed, n, corr, Theta, eta, test_data,\
                  n_layer, n_node, n_lr, n_epoch, patiences,\
                  n_layer_1, n_node_1, n_lr_1, n_epoch_1, patiences_1,\
                  n_layer_2, n_node_2, n_lr_2, n_epoch_2, patiences_2,\
                  maxloop=100,seq=0.01, m0=2):
    set_seed(Seed + i)
    warnings.filterwarnings("ignore")
    n_tr = int(0.8*n); n_va = int(0.2*n)
    train_data = generate_case_Deep(n=n_tr, corr=corr, Theta = Theta, eta = eta)
    val_data   = generate_case_Deep(n=n_va, corr=corr, Theta = Theta, eta = eta)
    Theta0 = [0,0]  #initial theta
    eta0 = np.mean(train_data['Z'])  # initial eta

    f_true = test_data['f_X']
    g_true = test_data['g_X']
    # deep
    res_d = CPDPLM(train_data,val_data,test_data,Theta0, eta0,\
            n_layer, n_node, n_lr, n_epoch, patiences,show_val=False,maxloop=maxloop,seq=seq)
    Theta_d = res_d['Theta']
    eta_d = res_d['eta']
    f_d = res_d['f_test']
    g_d = res_d['g_test']
    re_f_d = (np.sqrt(np.mean((f_d-np.mean(f_d)-f_true)**2)/np.mean(f_true**2))) #Re loss of g(x)
    re_g_d = (np.sqrt(np.mean((g_d-np.mean(g_d)-g_true)**2)/np.mean(g_true**2)))

    # additive (not recommend)
    # nodevec0 = np.array(np.linspace(0, 2, m0+2), dtype="float32")
    # res_a = CPALM(train_data,val_data,test_data,Theta0, eta0, m0, nodevec0, maxloop=maxloop,seq=seq)
    # Theta_a = res_a['Theta']
    # eta_a = res_a['eta']
    # f_a = res_a['f_test']
    # g_a = res_a['g_test']
    # re_f_a = (np.sqrt(np.mean((f_a-np.mean(f_a)-f_true)**2)/np.mean(f_true**2))) #Re loss of g(x)
    # re_g_a = (np.sqrt(np.mean((g_a-np.mean(g_a)-g_true)**2)/np.mean(g_true**2)))

    # linear
    res_l = CPLM(train_data,val_data,test_data,Theta0, eta0, maxloop=maxloop,seq=seq)
    Theta_l = res_l['Theta']
    eta_l = res_l['eta']
    f_l = res_l['f_test']
    g_l = res_l['g_test']
    re_f_l = (np.sqrt(np.mean((f_l-np.mean(f_l)-f_true)**2)/np.mean(f_true**2))) #Re loss of g(x)
    re_g_l = (np.sqrt(np.mean((g_l-np.mean(g_l)-g_true)**2)/np.mean(g_true**2)))


    # information for deep
    f_C_train = res_d['f_C_train']
    f_C_val = res_d['f_C_val']
    Z_train = train_data['Z']
    Z_val = val_data['Z']
    A1_train = train_data['A']
    A1_val = val_data['A']
    A2_train = train_data['A']*(Z_train>eta_d)
    A2_val = val_data['A']*(Z_val>eta_d)
    ## LFD for beta and gamma respectively, note as a1 and a2
    a1 = LFDCP(A1_train, A1_val, train_data,val_data,Theta_d,eta_d,f_C_train, f_C_val,\
            n_layer_1,n_node_1,n_lr_1,n_epoch_1,patiences_1,show_val = False)
    a2 = LFDCP(A2_train, A2_val, train_data,val_data,Theta_d,eta_d,f_C_train, f_C_val,\
            n_layer_2,n_node_2,n_lr_2,n_epoch_2,patiences_2,show_val = False)
    Info = np.zeros((2,2))
    Y_train = train_data['Y']
    X_train = train_data['X']
    A_train = train_data['A']
    AC_train = np.vstack((A_train,A_train*(Z_train>eta_d)))
    AC_train = AC_train.T
    epsilon_train = Y_train - AC_train@Theta_d -f_C_train
    Info[0,0] = np.mean(epsilon_train**2*(A1_train-a1)**2)
    Info[1,1] = np.mean(epsilon_train**2*(A2_train-a2)**2)
    Info[0,1] = np.mean(epsilon_train**2*(A1_train-a1)*(A2_train-a2))
    Info[1,0] = Info[0,1]
    Sigma = np.linalg.inv(Info)/n_tr
    # beta_deep
    se1_d = np.sqrt(Sigma[0,0])
    # gamma_deep
    se2_d = np.sqrt(Sigma[1,1])

    
    # information for linear
    
    train_val_data = {
        key: np.concatenate([train_data[key], val_data[key]], axis=0)
        for key in train_data
    }

    Z_train_val = train_val_data['Z']
    X_train_val = train_val_data['X']
    Y_train_val = train_val_data['Y']
    A_train_val = train_val_data['A']
    ind_l = (Z_train_val>eta_l)
    n_tv = A_train_val.shape[0]
    A_X_I = np.column_stack((A_train_val,A_train_val*ind_l,\
                             X_train_val,X_train_val*ind_l.reshape(n_tv,1)))
    p_AXI = A_X_I.shape[1]
    Info_l = (A_X_I.T@A_X_I+1e-8*np.identity(p_AXI))/n_tv
    Sigma_l = np.linalg.inv(Info_l)/n_tv
    # beta_deep
    se1_l = np.sqrt(Sigma_l[0,0])
    # gamma_deep
    se2_l = np.sqrt(Sigma_l[1,1])

    return Theta_d, eta_d, re_f_d, re_g_d, se1_d, se2_d, \
        Theta_l, eta_l, re_f_l, re_g_l, se1_l, se2_l




# An example, indicating the function one_case_deep.py
if __name__ == '__main__':
    # parameter 
    Theta = [-1,2]; eta = 2; corr = 0.5
    n = 1000; maxloop = 100; seq = 0.01; m0 = 2
    Seed = 1145; B = 20

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

    # Time record
    start_time = time.time()
    n_jobs = max(1, cpu_count() - 2)
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
    
    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds for estimation.")
    # Deep
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

    # Re
    # f
    RE_f_d = np.mean(np.array(re_f_d))
    SD_ref_d =  (np.sqrt(np.mean((np.array(re_f_d)-np.mean(np.array(re_f_d)))**2)))

    # g
    RE_g_d = np.mean(np.array(re_g_d))
    SD_reg_d =  (np.sqrt(np.mean((np.array(re_g_d)-np.mean(np.array(re_g_d)))**2)))

    # Linear
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

    # Re
    # f
    RE_f_l = np.mean(np.array(re_f_l))
    SD_ref_l =  (np.sqrt(np.mean((np.array(re_f_l)-np.mean(np.array(re_f_l)))**2)))

    # g
    RE_g_l = np.mean(np.array(re_g_l))
    SD_reg_l =  (np.sqrt(np.mean((np.array(re_g_l)-np.mean(np.array(re_g_l)))**2)))

    

    print(bias_b_d)
    print(bias_g_d)
    print(cp_b_d)
    print(cp_g_d)
    print(bias_eta_d)
    print(RE_f_d)
    print(SD_ref_d)
    print(RE_g_d)
    print(SD_reg_d)

    print(bias_b_l)
    print(bias_g_l)
    print(cp_b_l)
    print(cp_g_l)
    print(bias_eta_l)
    print(RE_f_l)
    print(SD_ref_l)
    print(RE_g_l)
    print(SD_reg_l)

    # result_linear5 = {
    #     'Theta': Theta_l,
    #     'eta': eta_l,
    #     'Re_f': re_f_l,
    #     'Re_g': re_g_l,
    #     'se1': se1_l,
    #     'se2': se2_l, 
    # }
    # filename = f"{"Est_d"}_{5}_{"n"}_{n}_{"Linear_2"}.npy"
    # np.save(filename, result_linear5, allow_pickle=True)
        
    # result_deep5 = {
    #     'Theta': Theta_d,
    #     'eta': eta_d,
    #     'Re_f': re_f_d,
    #     'Re_g': re_g_d,
    #     'se1': se1_d,
    #     'se2': se2_d, 
    # }
    # filename = f"{"Est_d"}_{5}_{"n"}_{n}_{"Deep_2"}.npy"
    # np.save(filename, result_deep5, allow_pickle=True)