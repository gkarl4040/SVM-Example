# Author: Tyler Karlsen
# Date: 06/20/2020
# Description: This script uses sklearn's decision tree library to classify
#              abalone as young (< 11 rings) or old (>= 11 rings) based on 
#              measurements of their physical characteristics.
#
#              Original dataset can be found here: https://archive.ics.uci.edu/ml/datasets/abalone

# Used for preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Used for generating model and tuning hyperparams
import warnings
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Read in dataset to data frame  
data = pd.read_csv('./abalone.data', sep=",", names=["Sex","Length",
                                                     "Diameter","Height",
                                                     "Whole Weight","Shucked Weight",
                                                     "Viscera Weight","Shell Weight",
                                                     "Rings"])

# Encode nominal values as integers
enc = LabelEncoder()
data['Sex'] = enc.fit_transform(data['Sex'])

# Drop target vector to get feature set
X = data.iloc[:, :8]

# Young abalone classed as -1, old abalone classed as +1
data.loc[data['Rings'] < 11, 'Rings'] = -1
data.loc[data['Rings'] >= 11, 'Rings'] = 1

# Store target vector
y = data.loc[:, 'Rings']

# Generate correlation matrix (drops signs)
corr_matrix = data.corr(method='pearson').abs()

# Get upper right triangle (the same as lower right, diagonal entries are self-correlated)
upper_right = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Get indices of dataframe columns with Pearson's coefficient > 0.90
strong_corr = [column for column in upper_right.columns if any(upper_right[column] > 0.90)]

# Drop columns with Pearson's coefficient > .90
data.drop(data[strong_corr], axis=1)

# Split dataset to get training/test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Hyperparams and kernels to be tested with cross-validation
param_list = [{'criterion': ['gini'],'max_depth': [1, 5, 10, 15, 20, 25, 32]},
              {'criterion': ['entropy'], 'max_depth': [1, 5, 10, 15, 20, 25, 32]}]

performance_metrics = ['precision', 'recall']

# Output performance metrics for optimal hyperparams
for metric in performance_metrics:
    print("--- Tuning hyperparameters for %s ---" % metric)
    
   # Suppress irrelevant warnings 
    warnings.filterwarnings('ignore') 
    
    # Perform grid search to find optimal hyperparams
    clf = GridSearchCV(tree.DecisionTreeClassifier(), param_list, scoring='%s_macro' % metric)
    clf.fit(X_train, y_train)

    #print("Optimal hyperparams on training set:")
    print(clf.best_params_)
    print()

    # Outputs performance metrics for model with optimized hyperparameters
    print("Performance metrics:")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()