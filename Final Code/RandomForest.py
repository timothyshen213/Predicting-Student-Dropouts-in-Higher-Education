import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

# Splitting data into 80% training and 20% testing
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# Create a pipeline with feature scaling and random forest classifier
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=10, random_state=0))

# Fit the pipeline to the training data
pipe.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = pipe.predict(X_test)

# Evaluate the accuracy of the model without cross-validation
print("Without CV: ", accuracy_score(Y_test, Y_pred))

# Evaluate the accuracy of the model with cross-validation
scores = cross_val_score(pipe, X_train, Y_train, cv=10)
print("With CV: ", scores.mean())

print("Precision Score: ", precision_score(Y_test, Y_pred,average='micro'))
print("Recall Score: ", recall_score(Y_test, Y_pred,average='micro'))
print("F1 Score: ", f1_score(Y_test, Y_pred,average='micro'))

param_grid = {
    'bootstrap': [False,True],
    'max_depth': [5, 10, 15, 20],
    'max_features': [4, 5, 6, None],
    'min_samples_split': [2, 10, 12],
    'n_estimators': [100, 200, 300]
}

rfc = RandomForestClassifier()

clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)

clf.fit(X_train, Y_train)
best_rfc = clf.best_estimator_

y_pred = best_rfc.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: ", accuracy)
print("Best Hyperparameters: ", clf.best_params_)

# Evaluate cross-validation score for the best model
cv_score = cross_val_score(best_rfc, X_train, Y_train, cv=10)
print("Cross-validation score: ", cv_score.mean())

print(classification_report(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))

# Visualizing results with a confusion matrix heatmap
cm = confusion_matrix(Y_test, y_pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='d', cmap='Blues')

ax.set_xlabel('Predictions')
ax.set_ylabel('Actuals')
ax.xaxis.set_ticklabels(['0', '1'])
ax.yaxis.set_ticklabels(['0', '1'])
plt.show()
