"""
This module runs a voting ensemble model to produce predictions for the Kaggle digit recognizer competition:
https://www.kaggle.com/c/digit-recognizer

Uses the sklearn wrapper for xgb: https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py

Example:
    nohup python -u ./digits_voting_ensemble_final.py > cmd.log &
"""
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import xgboost as xgb


#Load the training and test data
digits = pandas.read_csv("digits_train.csv")
digits_test = pandas.read_csv("digits_test.csv")
x_test = digits_test.astype(float).values
x_train = digits.iloc[:, 1:].astype(float).values
y_train = digits.iloc[:, 0].values

# Transfor the training and test data using a standardization scaler
stdc = StandardScaler()
x_std = stdc.fit_transform(x_train)
x_std_test = stdc.transform(x_test)

#Setup the models for the voting ensemble classifier
pipeLR = Pipeline([
    ('kernelPCA', KernelPCA(kernel='rbf', gamma=0.001400817988872654, n_components=614)),
    ('classifier', LogisticRegression(C=1000))
])

svc = SVC(gamma=.001, C=100, kernel='rbf', probability=True)
rf= RandomForestClassifier(n_estimators=400)
extraGini = ExtraTreesClassifier(n_estimators=400, criterion='gini')
extraEntropy = ExtraTreesClassifier(n_estimators=400, criterion='entropy')
xgb_model = xgb.XGBClassifier(nthread=2, silent=True, colsample_bytree=.4, learning_rate=0.05, max_depth=4, gamma=0, n_estimators=800)

#Setup the ensemble classifier
eclf = VotingClassifier(estimators=[
    ('lr', pipeLR),
    ('svcm', svc),
    ('rf', rf),
    ('extrag', extraGini),
    ('extraen', extraEntropy),
    ('xgbm', xgb_model)
    ], voting='soft')

print 'Starting training'
# Train the classifier
eclf.fit(x_std, y_train)
#Make predictions
result = eclf.predict(x_std_test)
#Save predictions
submission = pandas.DataFrame({"Label": result})
submission.index += 1
submission.to_csv(path_or_buf="voting_submission.csv", index_label="ImageId", index=True, header=True)
