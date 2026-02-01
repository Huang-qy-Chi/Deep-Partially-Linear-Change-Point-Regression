import numpy as np
import scipy.optimize as spo

#%%--------linear regression with given Theta and eta------------
def lnr_est(train_data,val_data,test_data,Theta,eta):
    '''
    Input: 
    m0: number of knot
    nodevec0: node vector of the B-spline
    Output:
    f_train_val: f prediction on X_train_val
    f_test: f prediction on X_test
    f_train_val: g prediction on X_train_val
    f_test: g prediction on X_test
    param: the parameter of the B-spline basises
    '''
    # Data combination
    train_val_data = {
        key: np.concatenate([train_data[key], val_data[key]], axis=0)
        for key in train_data
    }

    Z_train_val = train_val_data['Z']
    X_train_val = train_val_data['X']
    Y_train_val = train_val_data['Y']
    A_train_val = train_val_data['A']

    # Z_test = test_data['Z']
    X_test = test_data['X']
    # Y_test = test_data['Y']
    # A_test = test_data['A']
    p = X_train_val.shape[1]


    I = (Z_train_val>eta).astype(int)

    # loss function
    def GA(*args):
        b = args[0]
        I = (Z_train_val>eta).astype(int)
        f_est = np.dot(X_train_val, b[0: p]) 
        
        g_est = np.dot(X_train_val, b[(p):(2*p)]) 
        A_tvI = np.column_stack((A_train_val,A_train_val*I))
        model_add = A_tvI@Theta + f_est + g_est*I
        loss_fun = 0.5*(Y_train_val-model_add)**2
        return loss_fun.mean()
    
    # estimate f B-spline's parameter
    param = spo.minimize(GA,np.zeros(2*p),method='SLSQP')['x']
    
    # f(X) for train_val data
    f_train_val = np.dot(X_train_val, param[0:p]) 

    f_test = np.dot(X_test, param[0:p]) 
    
    # g(X) for train_val data
    g_train_val = np.dot(X_train_val, param[(p):(2*p)]) 

    g_test = np.dot(X_test, param[(p):(2*p)]) 

    return{
        'f_train_val': f_train_val,
        'f_test': f_test,
        'g_train_val': g_train_val,
        'g_test': g_test,
        'param': param
    }