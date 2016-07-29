"""
Performs a gridsearch to find the best parameters for xgb.
Available parameters are listed here:
    https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py

 nohup python -u ./digits_xgb_gridsearch.py > cmd.log &
"""
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import pandas
import warnings
warnings.filterwarnings("ignore")

#Load data
digits = pandas.read_csv("digits_train.csv", nrows=6000)
print digits.shape
x_train = digits.iloc[:, 1:].astype(float).values
y_train = digits.iloc[:, 0]

# You can add the nthreads parameter to use more threads
xgb_model = xgb.XGBClassifier()

#Dictionary of parameters to test
parameters = {
    'n_estimators': [100, 200, 300, 400, 600, 800],
    'learning_rate': [0.005, 0.05, 0.1, 0.3, .5, 1, 2],
    'max_depth': [4, 6, 8, 10],
    'subsample': [0.5, .75, 1],
    'colsample_bytree': [0.4, .6, .8, 1.0],
    'gamma':[0, .1, .5, 1]
}
#Run the grid search to find the best parameters.
# Change n_jobs for increased parallelism
clf = GridSearchCV(xgb_model, parameters, verbose=4, n_jobs=2)
clf.fit(x_train, y_train)


print '####################grid scores'
print clf.grid_scores_
print '####################best score'
print clf.best_score_
print '####################best parameters'
print clf.best_params_
print '####################best estimator'
print clf.best_estimator_


