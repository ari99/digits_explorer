"""
This module takes the output from setting gridsearch to verbosity level 3 or above, for example:
clf = GridSearchCV(xgb_model, parameters, verbose=4, n_jobs=2)

It parses the file created from gridsearch into a pandas dataframe and plots the scores in relation to
the parameters being tuned. This module is specific to the xgb parameters I test in the xgb_gridsearch_digits_official.py
file.

"""
import pandas as pd
import re
import numpy as np, pandas as pd; np.random.seed(0)
import seaborn as sns;
import matplotlib.pyplot as plt


def getNums(str1):
    """
        Parses a string for numbers.
        Args
            :param str1: string to parse
        Returns:
            :return: number found.
    """
    if pd.isnull(str1):
        return None
    return re.findall("[-+]?\d+[\.]?\d*", str1)[0]


def fixMins(series):
    """
        Fixes the column holding minutes.
        Sometimes it has "-" and the minutes are in a different columns.
        Args
            :param series: Original series.
        Returns:
            :return: Fixed series.
    """
    if series['mins1'] == '-':
        series['mins1'] = series['mins2']
    return series


def plotData(parameter_values, score_values, name, xticks_rang=None):
    """
    Makes a png file comparing parameter_values to score_values.
    Args:
        :param parameter_values: X axis values.
        :param score_values: Y axis values.
        :param name: Name of the parameter we are comparing
        :param xticks_rang: Range describing x ticks in plot.
    """
    plt.scatter(parameter_values, score_values)
    if xticks_rang is not None:
        plt.xticks(xticks_rang, rotation=45)
    plt.xlabel(name)
    plt.grid()
    # Create file
    plt.savefig(name, bbox_inches='tight')
    #plt.show()
    # Clear plot data
    plt.clf()


#Read and parse csv into a Pandas Dataframe.
scores = pd.read_csv("score.log", sep=' ',)
scores.columns = ['todrop1', 'todrop2', 'colsample', 'learnRate','estimators', 'subsample', 'depth', 'gamma', 'score', 'mins1', 'mins2']
scores = scores.drop(['todrop1', 'todrop2'], axis=1)


#Clean the data.
scores=scores.apply(fixMins, axis=1)
scores = scores.drop(['mins2'], axis=1)
scores = scores.applymap(getNums)
scores = scores.astype(float)

#Setup and plot data.
sns.set(style="white", color_codes=True)
range = np.arange(-.5, 2.5, .1)
plotData(scores['learnRate'], scores['score'], 'learnRate', range)
plotData(scores['colsample'], scores['score'], 'colsample')
plotData(scores['estimators'], scores['score'], 'estimators')
plotData(scores['subsample'], scores['score'], 'subsample')
plotData(scores['depth'], scores['score'], 'depth')
plotData(scores['gamma'], scores['score'], 'gamma')


