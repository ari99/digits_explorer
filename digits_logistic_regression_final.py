"""
Produces results for the digit recognizer Kaggle competition using Kernel Principal Component analysis and
Logistic Regression.
Competition: https://www.kaggle.com/c/digit-recognizer

Uses parameters found using digits_pipelines_searchcv_lr.py .

Example:
    nohup python -u ./digits_logistic_regression_final.py > cmd.log &
"""

import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA

#Load training and test data
digits = pandas.read_csv("digits_train.csv")
digits_test = pandas.read_csv("digits_test.csv")


x_train = digits.iloc[:, 1:].astype(float).values
y_train = digits.iloc[:, 0].values

x_test = digits_test.astype(float).values


#Setup the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kernelPCA', KernelPCA(kernel='rbf', gamma=0.001400817988872654, n_components=614)),
    ('classifier', LogisticRegression(random_state=1, C=1000))
])

#Train the model
pipe.fit(x_train, y_train)
#Make predictions
result = pipe.predict(x_test)

#Make the submission file
submission = pandas.DataFrame({"Label": result})
submission.index += 1
submission.to_csv(path_or_buf="lr_submission.csv", index_label="ImageId", index=True, header=True)

