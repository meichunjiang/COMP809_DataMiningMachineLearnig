'''
This lab will cover the important issue of pipelining.
Pipelining is a process that takes data from its raw form and subjects it to a series of transformations in order to
optimize a classification model for accuracy.

The processes involved in a pipeline may differ from situation to situation but typically a Data Scientist in Industry
implements the following processes:
1. Data Pre processing (feature selection, normalization, missing value imputation, outlier detection, etc).
2. Model building using a classifier
3. Parameter tuning – automatically determining the best values for a parameters using a tool such GridSearchCV.

Study the code provided below and run it in Python.

Extend the code to use the MLP Classifier instead of Logistic Regression and tune some of MLP’s critical parameters
such as the number of hidden neurons and the number of epochs.
'''

# Implementing a typical data mining workflow with the use of a pipeline
import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt

from numpy                      import interp
from sklearn.tree               import DecisionTreeClassifier
from sklearn.metrics            import confusion_matrix
from sklearn.metrics            import precision_score, recall_score, f1_score
from sklearn.metrics            import make_scorer
from sklearn.metrics            import roc_curve, auc
from sklearn.pipeline           import make_pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.decomposition      import PCA
from sklearn.preprocessing      import StandardScaler
from sklearn.preprocessing      import LabelEncoder
from sklearn.model_selection    import train_test_split
from sklearn.model_selection    import StratifiedKFold
from sklearn.model_selection    import cross_val_score
from sklearn.model_selection    import learning_curve
from sklearn.model_selection    import GridSearchCV

# Loading the Breast Cancer Wisconsin dataset
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

# Data preprocessing
print('header', df.head())
print(df.shape)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

# Assigning two dummy class labels 0 and 1
print(le.transform(['M', 'B']))

# dividing the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

# Defining a pipeline with standard scaler and decision tree
# DecisionTreeClassifier is the estimator and should be the last element of the pipeline
# Transformation in the pipeline should support to fit and transfom methods
# Estimator should support fit and predict methods
pipe_dt = make_pipeline(StandardScaler(),DecisionTreeClassifier(random_state=1 ))

#List of values for max_depth of decision tree
param_grid=[{'decisiontreeclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]

#Parameter optimization using grid search
#Check all the combinations of mentioned vales for the optimal performance. Tuning max_depth parameter
gs = GridSearchCV(estimator=pipe_dt,param_grid=param_grid, scoring='accuracy', cv=2)

# k-fold cross validation
# Training set is used for cross validation
# StratifiedKFold is a improved version of standard k-fold which gives better performance/good to deal with imbalance data sets
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train) #10-fold cross validation
scores = []
for k, (train, test) in enumerate(kfold):
    # use 9 folds from training set to train
    gs.fit(X_train[train], y_train[train])
    # remaing 1-fold to validate
    score = gs.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,np.bincount(y_train[train]), score))
    print(gs.best_params_)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=10)

# Finally take the average of accuracies as the final accuracy
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Choose the best model from grid search
clf = gs.best_estimator_

# Fitting and transforming training data through the pipeline and create the model
clf.fit(X_train, y_train)

# Feed testing data through the pipeline, transforming and do the prediction
y_pred = clf.predict(X_test)

# Looking at different performance evaluation metrics
# printing/displaying the confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig('06_09.png', dpi=300)
plt.show()

# Precision
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
# Recall
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))