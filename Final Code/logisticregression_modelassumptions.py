'''
*   Algorithm: Logistic Regression Model Assumptions
'''

# import libraries
import numpy as np
import pandas as pd
import seaborn as sns

## Check for Multicollinearity: Correlation Matrix
corr_matrix = X.corr()
sns.heatmap(corr_matrix, cmap='coolwarm')

# Drop Correlated Variables
dropout = dropout.drop(['Curricular units 1st sem (enrolled)',"Curricular units 2nd sem (enrolled)","Curricular units 1st sem (without evaluations)","Curricular units 2nd sem (without evaluations)","Curricular units 2nd sem (evaluations)","Curricular units 1st sem (evaluations)"],axis=1)

## Check for Linearity
# continuous variables
cont = ["Previous qualification (grade)","Admission grade","Curricular units 1st sem (grade)","Curricular units 2nd sem (grade)","Unemployment rate","Inflation rate","GDP"]
# Plot for Each Continuous Variable
for i in range(7):
    plt.figure()
    gre= sns.regplot(x= cont[i], y= "Target", data= dropout, logistic= True).set_title("Log Odds Linear Plot")
