"""
    This files contains functions to deploy three traditional machine learning models and to get misclassified data by each model.
"""

import sys
sys.path.append("../")

from utils import *
from config import config


def deploy_model(model_name):
    """
    Parameters
    ----------
    model_name : str
        Valid input choices are "LR", "SVM", and "NB", standing for Logistic Regression, 
        Support Vector Machine, and Naive Bayes, respectively. Function will immediately
        return if an invalid input is received.
    """    
    if (model_name != "SVM" and model_name != "LR" and model_name != "NB"):
        print("Model Name entered is invalid. There are three valid input choices: \"LR\", \"SVM\", \"NB\". They stand for Logistic Regression, Support Vector Machine, and Naive Bayes, respectively.")
        return
    else:
        df = pd.read_csv("../data/processed/final_clean.csv")
        train_test_model(model_name, df, config["save_model"], config["use_clean"], config["use_date"])

def get_misclassified_data(model_name):
    """
    Parameters
    ----------
    model_name : str
        Valid input choices are "LR", "SVM", and "NB", standing for Logistic Regression, 
        Support Vector Machine, and Naive Bayes, respectively. Function will immediately
        return if an invalid input is received.

    Return
    ----------
    misclassified_df : pandas.DataFrame
        subset of original DataFrame that is misclassified by the specified model
    ----------
    """  
    if (model_name != "SVM" and model_name != "LR" and model_name != "NB"):
        print("Model Name entered is invalid. There are three valid input choices: \"LR\", \"SVM\", \"NB\". They stand for Logistic Regression, Support Vector Machine, and Naive Bayes, respectively.")
        return
    else:
        df = pd.read_csv("../data/processed/final_clean.csv")
        misclassified_df = get_misclassified(model_name, df, config["save_misclassified"])
        return misclassified_df