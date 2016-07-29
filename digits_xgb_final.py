"""
Produces results for the digit recognizer kaggle competition using extreme gradient boosting.
Competition: https://www.kaggle.com/c/digit-recognizer

Uses the sklearn wrapper for xgb:
    https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
    https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py
Example:
    nohup python -u ./digits_xgb_final.py > cmd.log &
"""

import xgboost as xgb
import pandas

#Load training and testing data
digits = pandas.read_csv("digits_train.csv")
digits_test = pandas.read_csv("digits_test.csv")

x_train = digits.iloc[:, 1:].astype(float).values
y_train = digits.iloc[:, 0]
x_test = digits_test.astype(float).values

#Make XGB model using parameters found through a grid search
xgb_model = xgb.XGBClassifier(nthread=7, silent=False, colsample_bytree=.4, learning_rate=0.05, max_depth=4, gamma=0, n_estimators=800)
xgb_model.fit(x_train, y_train)


predictions = xgb_model.predict(x_test)


#Create submission file
submission = pandas.DataFrame({"Label": predictions})
submission.index += 1
submission.to_csv(path_or_buf="xgb_sub.csv", index_label="ImageId", index=True, header=True)

