"""
    This file contains code for the data visualization procedures.
"""

import pickle
import pandas as pd

import sys
sys.path.append("../")

from config import config
from utils import *

def data_cleanup():

    """
        Clean up the final dataset so that it is ready for visualization (and modeling in the future).
    """
    # Load data
    final_clean = pd.read_csv("../data/processed/final.csv")

    # Apply the preprocessing procedure defined in utils.py
    final_clean["clean_content"] = final_clean["content"].apply(preprocess_text)

    # Save the clean dataset to a new file
    if config["save_pickle"]:   
        with open("../data/processed/final_clean", "wb") as final_clean_file:
            pickle.dump(final_clean, final_clean_file)

    if config["save_csv"]:
        final_clean.to_csv("../data/processed/final_clean.csv", index = False)

    return final_clean

