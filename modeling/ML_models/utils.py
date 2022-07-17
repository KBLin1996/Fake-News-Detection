"""
This file contains helper functions in the modeling process.
"""

import pickle
import warnings
import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC 


# A function to vectorize features
def prep_features(df, save, use_clean, use_date): 
    """
    Parameters
    ----------
    df : pandas.DataFrame
        the DataFrame from which text input will be extracted
    
    save : boolean
        whether or not to save the vectorizer after fitting
    
    use_clean : boolean
        use clean text as input (i.e., preprocessed text that has been lemmatized, stemmed, etc.)
    
    use_date : boolean
        whether or not to include date as an input feature
    ----------

    Return
    ----------
    features : numpy.ndarray
        finalized features that can be fed into the model
    ----------
    """
    if use_clean:
        text = df["clean_content"]
    else:
        text = df["content"]

    # Vectorizing text input
    vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf = True, norm = 'l2', ngram_range = (1, 2))
    text_processed = vectorizer.fit_transform(text).toarray()
    
    if (save):
        with open("vectorizer", "wb") as f:
            pickle.dump(vectorizer, f)

    if use_date:
        # One-Hot Encoding to process Dates
        date_processed = LabelBinarizer().fit_transform(df.date)
        features = np.concatenate((date_processed, text_processed), axis = 1)
    else:
        features = text_processed

    return features


# Finalize training and testing data
def prep_data(df, use_clean, use_date):
    """
    Parameters
    ----------
    df : pandas DataFrame
        the DataFrame whose data will be used for training and testing
    
    use_clean : boolean
        use clean text as input (i.e., preprocessed text that has been lemmatized, stemmed, etc.)
    
    use_date : boolean
        whether or not to include date as an input feature
    ----------

    Return
    ----------
    X_train, X_test, y_train, y_test : numpy.ndarray
        standard train/test data split
    ----------    
    """
    X = prep_features(df, use_clean, use_date)
    Y = df["reliability"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    return X_train, X_test, y_train, y_test


# A helper function to train and test model
# Printing to console test statistics and confusion matrix
def train_test_model(model_name, df, save, use_clean, use_date):
    """
    Parameters
    ----------
    model_name : str
        name of the model; possible choices are "LR" indicating Logistic Regression,
        "SVM" indicating Support Vector Machine, and "NB" indicating Naive Bayes
    
    df : pandas DataFrame
        the DataFrame whose data will be used for training and testing
    
    save: boolean
        whether or not to save the trained model as a pickle file
    
    use_clean : boolean
        use clean text as input (i.e., preprocessed text that has been lemmatized, stemmed, etc.)
    
    use_date : boolean
        whether or not to include date as an input feature
    ----------    
    """
    X_train, X_test, y_train, y_test = prep_data(df, use_clean, use_date)

    if (model_name == "LR"):
        model = generate_model_LR(X_train, y_train)
    
    if (model_name == "SVM"):
        model = generate_model_SVM(X_train, y_train)

    if (model_name == "NB"):
        model = generate_model_NB(X_train, y_train)

    model.fit(X_train, y_train)

    if (save):
        with open("model_instances/" + model_name, "wb") as f:
            pickle.dump(model, f)

    ytest = np.array(y_test)
    predictions = model.predict(X_test)

    # confusion matrix and classification report(precision, recall, F1-score)
    print("Model Accuracy: " + str(accuracy_score(ytest, predictions)))  
    print("Precision Score: " + str(precision_score(ytest, predictions)))
    print("Recall Score: " + str(recall_score(ytest, predictions)))
    print("F1 Score: " + str(f1_score(ytest, predictions)))

    print("Confusion Matrix: ")
    print(confusion_matrix(ytest, predictions))


# Fine Tune Logistic Regression Model
def generate_model_LR(X_train, y_train):
    """
    Parameters
    ----------
    X_train: pandas.Series
        training input
    
    Y_train: pandas.Series
        training output
    ----------

    Return
    ----------
    result.best_estimator : model
        fine-tuned Logistic Regression model
    ----------
    """    
    warnings.filterwarnings("ignore")

    grid = {
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C":  [1e-3, 10],
        "solver": ["newton-cg", "lbfgs", "liblinear"]
    }
    cv = StratifiedKFold(n_splits=5)
    search = GridSearchCV(LogisticRegression(), grid, scoring = "accuracy", n_jobs = -1, cv = cv)
    result = search.fit(X_train, y_train)

    return result.best_estimator_


# Fine Tune Support Vector Machine Model
def generate_model_SVM(X_train, y_train):
    """
    Parameters
    ----------
    X_train: pandas Series
        training input
    
    Y_train: pandas Series
        training output

    Return
    ----------
    result.best_estimator : model
        fine-tuned Support Vector Machine model
    ----------
    """    
    warnings.filterwarnings("ignore")

    grid = {
        'C': [1, 0.01],
        'gamma': [1, 0.1, "scale"],
        'kernel': ['rbf', "linear"]
    }
    cv = StratifiedKFold(n_splits=5)
    search = GridSearchCV(SVC(), grid, scoring = "accuracy", n_jobs = -1, cv = cv)
    result = search.fit(X_train, y_train)

    return result.best_estimator_


# Fine Tune Naive Bayes Model
def generate_model_NB(X_train, y_train):
    """
    Parameters
    ----------
    X_train: pandas Series
        training input
    
    Y_train: pandas Series
        training output
    
    Return
    ----------
    result.best_estimator : model
        fine-tuned Naive Bayes model
    ----------
    """    
    warnings.filterwarnings("ignore")

    grid = {
        'var_smoothing': [1e-13, 1e-12, 1e-11, 1e-10, 1e-9]
    }
    cv = StratifiedKFold(n_splits=5)
    search = GridSearchCV(GaussianNB(), grid, scoring = "accuracy", n_jobs = -1, cv = cv)
    result = search.fit(X_train, y_train)
    
    return result.best_estimator_


# Get misclassified data from the testing dataset.  
def get_misclassified(model_name, df, save):
    """
    Parameters
    ----------
    model_name : str
        name of the model; possible choices are "LR" indicating Logistic Regression,
        "SVM" indicating Support Vector Machine, and "NB" indicating Naive Bayes
    
    df : pandas DataFrame
        the DataFrame whose data will be used for training and testing
    
    save: boolean
        whether or not to save the misclassified data into a csv file locally
    
    Return
    ----------
    misclassified_df : pandas.DataFrame
        subset of original DataFrame that is misclassified by the specified model
    ----------
    """  
    X = prep_features(df, False, False)
    Y = df["reliability"]

    with open("model_instances/" + model_name, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    misclassified = np.where(Y != predictions)
    misclassified_df = df.iloc[misclassified]

    if (save):
        misclassified_df.to_csv("misclassified_data/" + model_name + ".csv", index=False)

    return misclassified_df