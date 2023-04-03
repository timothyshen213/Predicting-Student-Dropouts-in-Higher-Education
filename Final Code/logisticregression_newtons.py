'''
*   Algorithm: Logistic Regression with Newton's Method
*   Code Partially Adapted From: Jia Rui Ong
*   Source: https://github.com/jrios6/Math-of-Intelligence/blob/master/2-Second_Order_Optimization/Logistic%20Regression%20with%20Newton's%20Method.ipynb
*   Authored Date: 2017
'''
## This is our second method for running Logistic Regression. 
## We first optimized by running a second-order optimization iterative technique of Newton's Method
## At each iteration it finds the Inverse Hessian Matrix.
## Thus will converge significantly faster when it finds f'=0
## To further optimize, we employed matrix decomposition in computing the Hessian Inverse
## We added LU/QR/Cholesky Decomposition method into our algorithm to do so
## Then tested to see witch one is optimal

# import libraries
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix

## Newton's Method Algorithm

# logistic function
def logfun(x):
    p = 1/(1+np.exp(-x)) # probability
    return p

# Updating beta Vector function
def newton_method_lrstep(beta0,y,X, method):
    betaX = np.dot(X,beta0) # beta*X
    yx = logfun(betaX) # y_hat
    y_hat = np.array(yx,ndmin=2) # converts to y_hat array
    gradient = np.dot(X.T, (y-y_hat)) # gradient

    ny_hat = 1-y_hat # 1-y_hat
    d = np.dot(y_hat,ny_hat.T) # finds D
    diag = np.diag(d) # computes diag of D
    d = np.diag(diag)
    hessian = X.to_numpy().T.dot(d).dot(X.to_numpy()) # hessian matrix
    n = hessian.shape[1] # ncol of hessian matrix
    # LU Decomposition to find Inverse
    if method == "LU":
        hessian_inv = np.linalg.inv(hessian)
    # QR Decomposition to find Inverse
    elif method == "QR":
        Q,R = np.linalg.qr(hessian)
        Rinv = linalg.solve_triangular(R,np.identity(n))
        hessian_inv = np.dot(Rinv,Q.T)
    # Cholesky Decompsition to find Inverse
    else:
        c, low = linalg.cho_factor(hessian)
        hessian_inv = linalg.cho_solve((c,low),np.identity(n))
    
    gd = np.dot(hessian_inv,gradient) # finds the step direction

    beta = beta0 + gd # updates coefficients

    return beta # new vector

# Checks Convergence
def tolerance_check(beta0, beta, eps):
    diff = np.abs(beta0-beta) # norm 
    if np.any(diff>eps): # if norm crosses threshold
        return False
    else:
        return True

# Logistic Regression via Newton's Method
def newton_method_logreg(beta0, y, X,eps,method):
    iterations = 0 # initial iterations
    converge = False # sets converge to false
    while not converge: # while converge is false
        beta = newton_method_lrstep(beta0,y,X,method) # finds new beta
        converge = tolerance_check(beta0,beta,eps) # checks convergence
        beta0 = beta # updates beta
        iterations +=1 # updates iterations
        print ("Iteration: {}".format(iterations))
    return beta

# Predictive Accuracy Function for Logistic Regression
def predict_lr(X,y,beta):
    xbeta = np.dot(X,beta) # x*beta
    predict = np.array(logfun(xbeta)) # predicted y
    threshold = 0.5*np.ones((predict.shape[0],1)) # sets 0.5 threshold
    pred_class = np.greater(predict,threshold) # converts y to 0,1
    accuracy = np.count_nonzero(np.equal(pred_class, y))/pred_class.shape[0] # checks accuracy
    return accuracy

## Testing Under True Dataset
accuracy = [] 
timeLU = []
timeQR = []
timeChol = []

# 100 Iterations with a Random Split of Training/Test Dataset
for i in range(100):
    # Splits data randomly
    X_train,x_test,y_train, y_test = train_test_split(X,y ,random_state = 10+i, test_size=0.20, shuffle=True)
    n = X.shape[1] # column of X
    beta0 = np.zeros((n,1)) # initial vector
    eps = 10**(-3) # set convergence threshold

    start_timeLU = time.time() # start time
    # Newton's Method via LU Decomposition
    temp = newton_method_logreg(beta0,y_train,X_train,eps,method="LU")
    end_timeLU = time.time() # end time
    timediffLU = end_timeLU-start_timeLU # time difference
    timeLU.append(timediffLU)

    start_timeQR = time.time() # start time
    # Newton's Method via QR Decomposition
    temp = newton_method_logreg(beta0,y_train,X_train,eps,method="QR") 
    end_timeQR = time.time() # end time
    timediffQR = end_timeQR-start_timeQR # time difference
    timeQR.append(timediffQR)

    start_timeChol = time.time() # start time
    # Newton's Method via Cholesky Decomposition
    temp = newton_method_logreg(beta0,y_train,X_train,eps,method="Chol") 
    end_timeChol = time.time() # end time
    timediffChol = end_timeChol-start_timeChol # time difference
    timeChol.append(timediffChol)

    acc = predict_lr(x_test,y_test,temp)
    accuracy.append(acc)

## Plotting Confusion Matrix
conf_matrix = metrics.confusion_matrix(y_test, predict) # confusion matrix
# Plotting Confusion Matrix
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

## Testing Under Simulated Data

# Generated Dataset, 10000 instances
X,y = make_classification(n_samples=10000, n_features=36, n_informative=36, 
                    n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,
                          class_sep=1.5)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Generated Dataset, 50000 instances
X,y = make_classification(n_samples=50000, n_features=36, n_informative=36, 
                    n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,
                          class_sep=1.5)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

# note: same code as true dataset to iterate, loop ran for each generated dataset above
for i in range(100):
    X_train,x_test,y_train, y_test = train_test_split(X,y ,random_state = 10+i, test_size=0.20, shuffle=True)
    n = X.shape[1]
    beta0 = np.zeros((n,1))  
    eps = 10**(-3)

    start_timeLU = time.time()
    temp = newton_method_logreg(beta0,y_train,X_train,eps,method="LU")
    end_timeLU = time.time()
    timediffLU = end_timeLU-start_timeLU
    timeLU.append(timediffLU)

    start_timeQR = time.time()
    temp = newton_method_logreg(beta0,y_train,X_train,eps,method="QR")
    end_timeQR = time.time()
    timediffQR = end_timeQR-start_timeQR
    timeQR.append(timediffQR)

    start_timeChol = time.time()
    temp = newton_method_logreg(beta0,y_train,X_train,eps,method="Chol")
    end_timeChol = time.time()
    timediffChol = end_timeChol-start_timeChol
    timeChol.append(timediffChol)

    acc = predict_lr(x_test,y_test,temp)
    accuracy.append(acc)

np.mean(accuracy)
np.var(accuracy)
np.mean(timediffLU)
np.var(timediffLU)
np.mean(timediffQR)
np.var(timediffQR)
np.mean(timediffChol)
np.mean(timediffChol)