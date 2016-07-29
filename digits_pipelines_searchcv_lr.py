"""
Uses RandomizedSearchCV to find optimal parameters for kernel principal component analysis and
logistic regression.

The resulting parameters are used in digits_logistic_regression_final.py .
"""
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
import scipy
from sklearn.grid_search import RandomizedSearchCV

#Load data
digits = pandas.read_csv("digits_train.csv", nrows=4000)



x = digits.iloc[:, 1:].astype(float).values
y = digits.iloc[:, 0].values


#Setup pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kernelPCA', KernelPCA(kernel='rbf')),
    ('classifier', LogisticRegression(random_state=1))
])

#Setup parameter distribution
param_dist = {'classifier__C': [.0001, .001, .01, .1, 10, 100, 1000],
              'kernelPCA__gamma': scipy.stats.expon(scale=.001),
              'kernelPCA__n_components':scipy.stats.expon(scale=100, loc=300)}


#Number of randomized searches
n_iter_search = 20

random_search = RandomizedSearchCV(pipe, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(x, y)
print '####################grid scores'
print random_search.grid_scores_
print '####################best score'
print random_search.best_score_
print '####################best params'
print random_search.best_params_
print '####################best estimator'
print random_search.best_estimator_
'''
Sample result:
####################grid scores
[mean: 0.66925, std: 0.00654, params: {'classifier__C': 0.01, 'kernelPCA__gamma': 0.0010781047443518006, 'kernelPCA__n_components': 630.5378463629321}, mean: 0.77725, std: 0.00959, params: {'classifier__C': 10, 'kernelPCA__gamma': 2.9483077301230337e-06, 'kernelPCA__n_components': 458.21926528781614}, mean: 0.65750, std: 0.00834, params: {'classifier__C': 0.01, 'kernelPCA__gamma': 0.0006586662687404317, 'kernelPCA__n_components': 536.7296202361442}, mean: 0.87450, std: 0.00545, params: {'classifier__C': 10, 'kernelPCA__gamma': 0.0002483889944314083, 'kernelPCA__n_components': 328.6713837314895}, mean: 0.88800, std: 0.00524, params: {'classifier__C': 10, 'kernelPCA__gamma': 0.0011039964224541206, 'kernelPCA__n_components': 354.34291939429227}, mean: 0.89600, std: 0.00409, params: {'classifier__C': 1000, 'kernelPCA__gamma': 0.0005975169482613954, 'kernelPCA__n_components': 594.126232047368}, mean: 0.87275, std: 0.00721, params: {'classifier__C': 100, 'kernelPCA__gamma': 4.279881198834678e-05, 'kernelPCA__n_components': 327.67405848453507}, mean: 0.75725, std: 0.00794, params: {'classifier__C': 0.1, 'kernelPCA__gamma': 0.00020259122534627076, 'kernelPCA__n_components': 304.779646600866}, mean: 0.26175, std: 0.00750, params: {'classifier__C': 0.0001, 'kernelPCA__gamma': 0.0006408617932834473, 'kernelPCA__n_components': 335.35793337344046}, mean: 0.40875, std: 0.00330, params: {'classifier__C': 0.001, 'kernelPCA__gamma': 0.0010546539339537505, 'kernelPCA__n_components': 493.9738520221855}, mean: 0.79100, std: 0.00193, params: {'classifier__C': 0.1, 'kernelPCA__gamma': 0.000871883151095226, 'kernelPCA__n_components': 521.4960451342208}, mean: 0.88875, std: 0.00701, params: {'classifier__C': 10, 'kernelPCA__gamma': 0.0005787070866024719, 'kernelPCA__n_components': 368.3141647410449}, mean: 0.89175, std: 0.00326, params: {'classifier__C': 1000, 'kernelPCA__gamma': 0.001219789615023041, 'kernelPCA__n_components': 349.32880405972594}, mean: 0.90350, std: 0.00324, params: {'classifier__C': 1000, 'kernelPCA__gamma': 0.001400817988872654, 'kernelPCA__n_components': 614.2717025105088}, mean: 0.38200, std: 0.00461, params: {'classifier__C': 0.001, 'kernelPCA__gamma': 0.0006795977488587178, 'kernelPCA__n_components': 444.5340782772605}, mean: 0.18100, std: 0.00540, params: {'classifier__C': 0.0001, 'kernelPCA__gamma': 0.0031861302136234225, 'kernelPCA__n_components': 426.8729719538092}, mean: 0.29300, std: 0.00596, params: {'classifier__C': 0.001, 'kernelPCA__gamma': 0.003173883016376816, 'kernelPCA__n_components': 336.84682244027226}, mean: 0.12600, std: 0.00379, params: {'classifier__C': 0.001, 'kernelPCA__gamma': 0.00013636718478863275, 'kernelPCA__n_components': 322.8184961144575}, mean: 0.89450, std: 0.00063, params: {'classifier__C': 100, 'kernelPCA__gamma': 0.0025340842608303294, 'kernelPCA__n_components': 426.913390414986}, mean: 0.88425, std: 0.00441, params: {'classifier__C': 1000, 'kernelPCA__gamma': 0.0008062486510183742, 'kernelPCA__n_components': 305.47228639920075}]
####################best score
0.9035
####################best params
{'classifier__C': 1000, 'kernelPCA__gamma': 0.001400817988872654, 'kernelPCA__n_components': 614.2717025105088}
####################best estimator
Pipeline(steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('kernelPCA', KernelPCA(alpha=1.0, coef0=1, degree=3, eigen_solver='auto',
     fit_inverse_transform=False, gamma=0.00140081798887, kernel='rbf',
     kernel_params=None, max_iter=None, n_components=614.271702511,
     rem...nalty='l2', random_state=1, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False))])
'''