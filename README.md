# Deep Partially Linear Change-Point Regression
This code mainly focuses on estimating a partial linear model with a change point:
$$Y=A^\top\beta+f(X)+(A^\top\gamma+g(X))I(Z>\eta)+\epsilon,$$
where: 
1. $Y\in \mathbb{R}$ represents the response,
2. $A\in \mathbb{R}^p$ means the covariate (treatment) with linear effects,
3. $X\in \mathbb{R}^r$ denotes other covariates estimated by the DNN,
4. $I(\cdot)$ means the indicator function, $f,g: \mathbb{R}^r\to \mathbb{R}$ are multivariate functions $(usually \ r\geq 3)$,
5.  $Z\in \mathbb{R}$ represents the change-point covariate.

An additional impact $A_i^\top\gamma+g(X_i)$ will occur if the $i$ th observation satisfies $Z_i>\eta$, where $\eta\in (\min(Z),\max(Z))$ 
is an unknown threshold.
The function inputs data $(Y_i,A_i,X_i,Z_i)$ and output the estimation of the parameter $\theta=(\beta,\gamma)$, the change-point
$\eta$ and the estimation of two functions $(f,g)$ via a profile estimation procedure. 

Readers can attach to the file $\textsf{example.ipynb}$, which automatically imports functions in $\textbf{\textsf{.py}}$, and it reports:
1. the bias, SSE, ESE and CP (close to 0.95) of $\theta=(\beta,\gamma)$,
2. the bias of $\eta$,
3. the relative error (RE) of $(f,g)$ on test data.

To avoid overfitting, the hyperparameters n_lr (learning rate), n_node (width), n_layer (depth), n_epoch (max epoch)
and patiences (when to early stop) need to be adjusted, and the grid for $\eta$ grid search can be adjusted 
by seq (initial=0.01).

The code recommends Python version>=3.13.2, Numpy >=2.2.4, Pytorch>=2.7.0 and Scipy>=1.15.2. 

Copyright © 2025 Q. Huang. All rights reserved. 

13/08/2025, Hung Hom, Kowloon, Hong Kong, China.














