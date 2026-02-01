#%%
import numpy as np
import torch 
from spl_estimate import spl_est
from Theta_estimate import Theta_est
from eta_estimate import eta_est

#%%-------------------------Estimation by iteration-----------------------------
def CPALM(train_data,val_data,test_data,Theta0, eta0,m0, nodevec0 , maxloop=100, seq = 0.01):
    # index for whether it converges
    C_index = 0
    
    # Data combination 
    train_val_data = {
        key: np.concatenate([train_data[key], val_data[key]], axis=0)
        for key in train_data
    }

    Z_train_val = train_val_data['Z']
    X_train_val = train_val_data['X']
    Y_train_val = train_val_data['Y']
    A_train_val = train_val_data['A']

    # iterate estimation
    for i in range(maxloop):
        # B-spline estimation
        spl_res = spl_est(train_data, val_data, test_data,Theta0,eta0,m0,nodevec0)
        f_train_val = spl_res['f_train_val']
        g_train_val = spl_res['g_train_val']
        f_test = spl_res['f_test']
        g_test = spl_res['g_test']

        # Parameter estimation
        Theta = Theta_est(Y_train_val, A_train_val, Z_train_val, f_train_val, g_train_val, eta0)

        # Change point estimation
        eta = eta_est(Y_train_val, A_train_val, Z_train_val, f_train_val, g_train_val, Theta, seq = seq)

        # whether stop the loop
        if (np.max(abs(Theta0-Theta)) <= 0.01):
            C_index = 1
            break
        Theta0 = Theta
        eta0 = eta
    
    return{
        'C_index': C_index, 
        'Theta': Theta, 
        'eta': eta,
        'f_test': f_test,
        'g_test': g_test
    }

if __name__ == '__main__':
    import numpy as np
    import numpy.random as ndm
    import matplotlib.pyplot as plt
    from seed import set_seed
    from data_generator import generate_case_Deep

    # True value
    corr = 0.5
    Theta = [-1,2]
    eta = 2

    # Parameter
    m0 = 2 
    nodevec0 = np.array(np.linspace(0, 2, m0+2), dtype="float32")

    set_seed(3407)
    test_data = generate_case_Deep(500,corr,Theta,eta)
    X_test = test_data['X']
    Z_test = test_data['Z']
    A_test = test_data['A']
    Y_test = test_data['Y']
    f_true = test_data['f_X']
    g_true = test_data['g_X']
    Res_true = test_data['f_X_C']

    n = 1000; n_tr = int(0.8*n); n_va = int(0.2*n)
    set_seed(1145)

    # Generate data
    train_data = generate_case_Deep(n_tr,corr,Theta,eta)
    val_data = generate_case_Deep(n_va,corr,Theta,eta)

    # Initial values
    Theta0 = [0,0]  #initial theta
    eta0 = np.mean(train_data['Z'])  # initial eta

    # Estimation
    res = CPALM(train_data,val_data,test_data,Theta0, eta0, m0, nodevec0, maxloop=100,seq=0.01)
    
    # Result
    # record the results
    Theta_res = res['Theta'] # vector to add row by row
    eta_res = res['eta'] # change point   
    # test data to calculate Re and Sd_Re for g and h
    f_test_res = res['f_test'] 
    g_test_res = res['g_test']
    print('-------------------------------------------------')
    print('Estimation for reg para and change point para: ')
    print('Theta: ', Theta_res)
    print('eta: ', eta_res)
    print('-------------------------------------------------')