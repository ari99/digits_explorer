"""
This module tests the accuracy of various models to see if we want to include them in a voting ensemble module.
 It outputs the individual module accuracy as well as the voting ensemble accuracy.
 Produces results for the digit recognizer kaggle competition: https://www.kaggle.com/c/digit-recognizer
Uses the sklearn wrapper for xgb: https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
 Example:
    nohup python -u ./digits_voting_ensemble_explore.py > cmd.log &
"""
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from sklearn import cross_validation

# Load data info a pandas dataframe
digits = pandas.read_csv("digits_train.csv", nrows=3000)

# Split the data
x = digits.iloc[:, 1:].astype(float).values
y = digits.iloc[:, 0].values

# Standardize the training data
stdc = StandardScaler()
x_std = stdc.fit_transform(x)


# Create a pipeline for LogisticRegression
pipeLR = Pipeline([
    ('kernelPCA', KernelPCA(kernel='rbf', gamma=0.001400817988872654, n_components=614)),
    ('classifier', LogisticRegression(C=1000))
])

# Setup other classifiers
svc = SVC(gamma=.001, C=100, kernel='rbf', probability=True)
rf= RandomForestClassifier(n_estimators=400)
extraGini = ExtraTreesClassifier(n_estimators=400, criterion='gini')
extraEntropy = ExtraTreesClassifier(n_estimators=400, criterion='entropy')
gnb = GaussianNB()
ada=AdaBoostClassifier(n_estimators=400, learning_rate=.1)

xgb_model = xgb.XGBClassifier(nthread=2, silent=True, colsample_bytree=.4, learning_rate=0.05, max_depth=4, gamma=0, n_estimators=800)

# Setup voting ensemble classifier
eclf = VotingClassifier(estimators=[
    ('lr', pipeLR),
    ('svcm', svc),
    ('rf', rf),
    ('extrag', extraGini),
    ('extraen', extraEntropy),
    ('xgbm', xgb_model),
    ('gnbm', gnb),
    ('adam', ada)
    ], voting='soft')

print 'starting loop'
# Output the accuracy for each model and the ensemble model
for clf, label in zip([
                        pipeLR,
                       svc,
                       rf,
                        extraGini,
                        extraEntropy,
                       xgb_model,
                       gnb,
                       ada,
                       eclf
                       ],
                      [
                        'Logistic Regression',
                       'SVC',
                       'Random Forest',
                          'Extra gini',
                          'Extra entropy',
                          'xgb',
                          'GaussianNB',
                          'Adaboost',
                          'Ensemble'
                      ]):
    print label
    scores = cross_validation.cross_val_score(clf, x_std, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))


'''

Logistic Regression
Accuracy: 0.9037 (+/- 0.0124) [Logistic Regression]
SVC
Accuracy: 0.9183 (+/- 0.0125) [SVC]
Random Forest
Accuracy: 0.9290 (+/- 0.0060) [Random Forest]
Extra gini
Accuracy: 0.9357 (+/- 0.0066) [Extra gini]
Extra entropy
Accuracy: 0.9340 (+/- 0.0068) [Extra entropy]
xgb
Accuracy: 0.9310 (+/- 0.0051) [xgb]
GaussianNB
Accuracy: 0.5520 (+/- 0.0102) [GaussianNB]
Adaboost
Accuracy: 0.7113 (+/- 0.0245) [Adaboost]
Ensemble
Accuracy: 0.9360 (+/- 0.0072) [Ensemble]





Without Gaussian Naive Bayes and Adaboost:
Logistic Regression
Accuracy: 0.9037 (+/- 0.0124) [Logistic Regression]
SVC
Accuracy: 0.9183 (+/- 0.0125) [SVC]
Random Forest
Accuracy: 0.9297 (+/- 0.0080) [Random Forest]
Extra gini
Accuracy: 0.9360 (+/- 0.0070) [Extra gini]
Extra entropy
Accuracy: 0.9347 (+/- 0.0046) [Extra entropy]
xgb
Accuracy: 0.9320 (+/- 0.0051) [xgb]
Ensemble
Accuracy: 0.9397 (+/- 0.0077) [Ensemble]

'''

