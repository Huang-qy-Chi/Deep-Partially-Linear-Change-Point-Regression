import numpy as np
import scipy.optimize as spo
from B_spline3 import B_S

#%%--------B-Spline estimation with given Theta and eta------------
def spl_est(train_data,val_data,test_data,Theta,eta,m0,nodevec0):
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

    # B-spline basis for X_1,...,X_5
    B_0 = B_S(m0, X_train_val[:,0], nodevec0)
    B_1 = B_S(m0, X_train_val[:,1], nodevec0)
    B_2 = B_S(m0, X_train_val[:,2], nodevec0)
    B_3 = B_S(m0, X_train_val[:,3], nodevec0)
    B_4 = B_S(m0, X_train_val[:,4], nodevec0)

    I = (Z_train_val>eta).astype(int)

    # loss function
    def GA(*args):
        b = args[0]
        I = (Z_train_val>eta).astype(int)
        f_est = np.dot(B_0, b[0:(m0+4)]) + np.dot(B_1, b[(m0+4):(2*(m0+4))]) + np.dot(B_2, b[(2*(m0+4)):(3*(m0+4))]) \
            + np.dot(B_3, b[(3*(m0+4)):(4*(m0+4))]) + np.dot(B_4, b[(4*(m0+4)):(5*(m0+4))])\
                + b[5*(m0+4)]*np.ones(X_train_val.shape[0])
        
        g_est = np.dot(B_0, b[(5*(m0+4)+1):(6*(m0+4)+1)]) + np.dot(B_1, b[(6*(m0+4)+1):(7*(m0+4)+1)])\
            + np.dot(B_2, b[(7*(m0+4)+1):(8*(m0+4)+1)]) \
            + np.dot(B_3, b[(8*(m0+4)+1):(9*(m0+4)+1)]) + np.dot(B_4, b[(9*(m0+4)+1):(10*(m0+4)+1)])\
                + b[10*(m0+4)+1]*np.ones(X_train_val.shape[0])
        A_tvI = np.column_stack((A_train_val,A_train_val*I))
        model_add = A_tvI@Theta + f_est + g_est*I
        loss_fun = 0.5*(Y_train_val-model_add)**2
        return loss_fun.mean()
    
    # estimate f B-spline's parameter
    param = spo.minimize(GA,np.zeros(10*(m0+4)+2),method='SLSQP')['x']
    
    # f(X) for train_val data
    f_train_val = np.dot(B_0, param[0:(m0+4)]) + np.dot(B_1, param[(m0+4):(2*(m0+4))])\
        + np.dot(B_2, param[(2*(m0+4)):(3*(m0+4))]) + np.dot(B_3, param[(3*(m0+4)):(4*(m0+4))]) \
            + np.dot(B_4, param[(4*(m0+4)):(5*(m0+4))]) + param[5*(m0+4)]*np.ones(X_train_val.shape[0])

    f_test = np.dot(B_S(m0, X_test[:,0], nodevec0), param[0:(m0+4)]) \
        + np.dot(B_S(m0, X_test[:,1], nodevec0), param[(m0+4):(2*(m0+4))]) \
            + np.dot(B_S(m0, X_test[:,2], nodevec0), param[(2*(m0+4)):(3*(m0+4))]) \
                + np.dot(B_S(m0, X_test[:,3], nodevec0), param[(3*(m0+4)):(4*(m0+4))])\
                    + np.dot(B_S(m0, X_test[:,4], nodevec0), param[(4*(m0+4)):(5*(m0+4))])\
                        + param[5*(m0+4)]*np.ones(X_test.shape[0])
    
    # g(X) for train_val data
    g_train_val = np.dot(B_0, param[5*(m0+4)+1:6*(m0+4)+1]) + np.dot(B_1, param[6*(m0+4)+1:(7*(m0+4)+1)])\
        + np.dot(B_2, param[(7*(m0+4)+1):(8*(m0+4)+1)]) + np.dot(B_3, param[(8*(m0+4)+1):(9*(m0+4)+1)]) \
            + np.dot(B_4, param[(9*(m0+4)+1):(10*(m0+4)+1)]) + param[10*(m0+4)+1]*np.ones(X_train_val.shape[0])

    g_test = np.dot(B_S(m0, X_test[:,0], nodevec0), param[5*(m0+4)+1:6*(m0+4)+1]) \
        + np.dot(B_S(m0, X_test[:,1], nodevec0), param[6*(m0+4)+1:(7*(m0+4)+1)]) \
            + np.dot(B_S(m0, X_test[:,2], nodevec0), param[(7*(m0+4)+1):(8*(m0+4)+1)]) \
                + np.dot(B_S(m0, X_test[:,3], nodevec0), param[(8*(m0+4)+1):(9*(m0+4)+1)])\
                    + np.dot(B_S(m0, X_test[:,4], nodevec0), param[(9*(m0+4)+1):(10*(m0+4)+1)])\
                        + param[10*(m0+4)+1]*np.ones(X_test.shape[0])

    return{
        'f_train_val': f_train_val,
        'f_test': f_test,
        'g_train_val': g_train_val,
        'g_test': g_test,
        'param': param
    }