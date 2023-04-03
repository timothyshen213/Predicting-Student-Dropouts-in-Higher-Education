'''
*   Algorithm: Logistic Regression with Gradinet Descent
*   Code Partially Adapted From: Phong Hoangg,
*   Source: https://github.com/PhongHoangg/Gradient-Descent-for-Logistics-Regression/blob/main/Gradient%20Descent%20for%20Logistics%20Regression.ipynb
*   Authored Date: July 31, 2021
'''
## First method for Logistic Regression. 
## Maximizes loss function 
## Tested to see which learning rate is optimal in terms of accuray and time

# libraries
import time
import numpy as np
from math import e
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generateXvector(X):
    """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
        Parameters:
          X:  independent variables matrix
        Return value: the matrix that contains all the values in the dataset, not include the outcomes values 
    """
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def theta_init(X):
    """ Generate an initial value of vector θ from the original independent variables matrix
         Parameters:
          X:  independent variables matrix
        Return value: a vector of theta filled with initial guess
    """
    theta = np.random.randn(len(X[0])+1, 1)
    return theta

def sigmoid_function(X):
    """ Calculate the sigmoid value of the inputs
         Parameters:
          X:  values
        Return value: the sigmoid value
    """
    return 1/(1+e**(-X))

def tolerance_check(beta0, beta, eps):
    """ Check tolerance againts difference
         Parameters:
          beta0: initial value
          beta: gradient
          eps: tolerance limit for cheking difference between beta0  and beta
        Return value: True if the differnece is bigger than tolerance
    """
    diff = np.abs(beta0-beta) # absolute value
    if np.any(diff>eps):
        return False
    else:
        return True
    
def Logistics_Regression(X, y, learningrate, eps):
    """ Find the Logistics regression model for the data set
         Parameters:
          X: independent variables matrix
          y: dependent variables matrix
          learningrate: learningrate of Gradient Descent
          eps: tolerance limit for cheking difference
        Return value: the final theta vector and the plot of cost function
    """
    y_new = np.reshape(y, (len(y), 1))   
    cost_lst = []
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m =  X.shape[0]
    iterations = 0
    #for i in range(iterations):
    converge = False
    while not converge:
        gradients = 2/m * vectorX.T.dot(sigmoid_function(vectorX.dot(theta)) - y_new)
        theta1 = theta - learningrate * gradients
        y_pred = sigmoid_function(vectorX.dot(theta))
        cost_value = - np.sum(y_new*np.log(y_pred) + ((1-y_new)*np.log(1-y_pred)))/(len(y_pred))
        converge = tolerance_check(theta,theta1,eps) # checks convergence
        theta = theta1
        iterations += 1
    
        # Calculate loss for each training instance
        cost_lst.append(cost_value)
    
    return theta

#%% with cost function plot
def Logistics_Regression_plot_cost(X, y, learningrate, eps):
    """ Find the Logistics regression model for the data set
         Parameters:
          X: independent variables matrix
          y: dependent variables matrix
          learningrate: learningrate of Gradient Descent
          eps: tolerance limit for cheking difference
        Return value: the final theta vector and the plot of cost function
    """
    y_new = np.reshape(y, (len(y), 1))   
    cost_lst = []
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m =  X.shape[0]
    iterations = 0
    converge = False
    while not converge:
        gradients = 2/m * vectorX.T.dot(sigmoid_function(vectorX.dot(theta)) - y_new)
        theta1 = theta - learningrate * gradients
        y_pred = sigmoid_function(vectorX.dot(theta))
        cost_value = - np.sum(y_new*np.log(y_pred) + ((1-y_new)*np.log(1-y_pred)))/(len(y_pred))
        converge = tolerance_check(theta,theta1,eps) # checks convergence
        theta = theta1
        iterations += 1
    #Calculate the loss for each training instance
        cost_lst.append(cost_value)
    
    plt.figure()
    plt.plot(np.arange(1,iterations),cost_lst[1:], color = 'red')
    plt.title('Cost Function')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.xlim(0, iterations)
    return theta


#%% initial test

dropout = pd.read_csv('data.csv', header=0, sep=';')
dropout['Target'].replace(['Dropout', 'Graduate',"Enrolled"],[0, 1,1], inplace=True)

X = dropout.drop(['Target'],axis=1)
y = dropout[["Target"]]

X_train,X_test,y_train, y_test = train_test_split(X,y ,random_state=100, test_size=0.20, shuffle=True)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

eps = 10**(-3)
beta_init = Logistics_Regression(X_train,y_train, 1, eps)

#%% Testing - from online resource

def predict_lr(X,y,beta):
    xbeta = np.dot(X,beta)
    predict = np.array(sigmoid_function(xbeta))
    threshold = 0.5*np.ones((predict.shape[0],1))
    pred_class = np.greater(predict,threshold)
    accuracy = np.count_nonzero(np.equal(pred_class, y))/pred_class.shape[0]
    return accuracy, pred_class

acc, predict = predict_lr(X_test,y_test,beta_init[1:])

#%%
# from sklearn import metrics
# from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# conf_matrix = metrics.confusion_matrix(y_test, predict)
# fig, ax = confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()

# conf matrix TEST
cm = confusion_matrix(y_test, predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dropout', 'Graduate'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Test')
plt.show() 

# conf matrix TRAIN - maybe unecerasy
# cm = confusion_matrix(y_train, predict)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dropout', 'Graduate'])
# disp.plot(cmap='Blues')
# plt.title('Train')
# plt.show() 


#%% Learning rate (alpha) exploration

# initialize lists to store values
accuracy1 = []
accuracy75 = []
accuracy25 = [] 
accuracy5 = []
accuracy01 = []

time1 = []
time75 = []
time25 = []
time5 = []
time01 = []

#%% alpha = 1
for i in range(100):
    X_train,X_test,y_train, y_test = train_test_split(X,y ,random_state = 10+i, test_size=0.20, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    eps = 10**(-3)

    start_time1 = time.time()
    beta = Logistics_Regression(X_train,y_train, 1, eps)
    end_time1 = time.time()
    timediff1 = end_time1 - start_time1
    time1.append(timediff1)
    temp = beta[1:]
    acc = predict_lr(X_test,y_test,temp)
    accuracy1.append(acc[0])  


#%% alpha = 0.75
for i in range(100):
    X_train,X_test,y_train, y_test = train_test_split(X,y ,random_state = 10+i, test_size=0.20, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    eps = 10**(-3)

    start_time75 = time.time()
    beta = Logistics_Regression(X_train,y_train, 0.75, eps)
    end_time75 = time.time()
    timediff75 = end_time75 - start_time75
    time75.append(timediff75)
    temp = beta[1:]
    acc = predict_lr(X_test,y_test,temp)
    accuracy75.append(acc[0])  


#%% alpha = 0.5
for i in range(100):
    X_train,X_test,y_train, y_test = train_test_split(X,y ,random_state = 10+i, test_size=0.20, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    eps = 10**(-3)
    
    start_time5 = time.time()
    beta = Logistics_Regression(X_train,y_train, 0.5, eps)
    end_time5 = time.time()
    timediff5 = end_time5-start_time5
    time5.append(timediff5)
    temp = beta[1:]
    acc = predict_lr(X_test,y_test,temp)
    accuracy5.append(acc[0])


#%% alpha = 0.25
for i in range(100):
    X_train,X_test,y_train, y_test = train_test_split(X,y ,random_state = 10+i, test_size=0.20, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    eps = 10**(-3)
    
    start_time25 = time.time()
    beta = Logistics_Regression(X_train,y_train, 0.25, eps)
    end_time25 = time.time()
    timediff25 = end_time25-start_time25
    time25.append(timediff25)
    temp = beta[1:]
    acc = predict_lr(X_test,y_test,temp)
    accuracy25.append(acc[0])


#%% alpha = 0.01
for i in range(100):
    X_train,X_test,y_train, y_test = train_test_split(X,y ,random_state = 10+i, test_size=0.20, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    eps = 10**(-3)
    
    start_time01 = time.time()
    beta = Logistics_Regression(X_train,y_train, 0.01, eps)
    end_time01 = time.time()
    timediff01 = end_time01-start_time01
    time01.append(timediff01)
    temp = beta[1:]
    acc = predict_lr(X_test,y_test,temp)
    accuracy01.append(acc[0])



#%% 
# x axis values (for plotting)
x_axis_alpha = [0.01, 0.25, 0.5, 0.75, 1]

# convert to np array in order to find mean
acc1_np = np.array(accuracy1)
acc75_np = np.array(accuracy75)
acc5_np = np.array(accuracy5)
acc25_np = np.array(accuracy25)
acc01_np = np.array(accuracy01)

acc_list = [acc01_np, acc25_np, acc5_np, acc75_np, acc1_np]
# boxplot
plt.figure()
plt.boxplot(acc_list)
plt.title('Accuracy boxplot for different learning rates')
plt.show()

# list of average accurasies for the five different alphas
avg_acc_list = [acc01_np.mean(), acc25_np.mean(), acc5_np.mean(), acc75_np.mean(), acc1_np.mean()]

# plot ACCURACY
plt.figure()
plt.plot(x_axis_alpha, avg_acc_list, color = 'red')
plt.title('Average accuracy for each alpha value')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')

 
# convert to np array in order to find mean
time1_np = np.array(time1)
time75_np = np.array(time75)
time5_np = np.array(time5)
time25_np = np.array(time25)
time01_np = np.array(time01)

# list of average times for the five different alphas
avg_time_list = [time01_np.mean(), time25_np.mean(), time5_np.mean(), time75_np.mean(),time1_np.mean()]

# plot TIME
plt.figure()
plt.plot(x_axis_alpha, avg_time_list, color = 'red')
plt.title('Average time taken for each alpha value')
plt.xlabel('Alpha')
plt.ylabel('Time')


# plot COST
gd_01 = Logistics_Regression_plot_cost(X_train,y_train, 0.01, eps)
gd_25 = Logistics_Regression_plot_cost(X_train,y_train, 0.25, eps)
gd_5 = Logistics_Regression_plot_cost(X_train,y_train, 0.5, eps)
gd_75 = Logistics_Regression_plot_cost(X_train,y_train, 0.75, eps)
gd_1 = Logistics_Regression_plot_cost(X_train,y_train, 1, eps)

