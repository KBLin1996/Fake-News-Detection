"""
This file contains APIs used to clean up data in the data wrangling process.
"""

import pickle
import numpy as np
import pandas as pd
from datetime import date

########################## Shared Functions across Two Datasets ##########################

# To process DATE
def get_date(iso_str) -> date:   
    """
    Parameters
    ----------
    iso_str : str
        string in ISO format to be converted to a DateTime Object

    """

    # Only need first 10 digits of date (MM-DD-YYYY).
    iso_str = iso_str[0:10]
    # Error Handling to catch messy data that does not follow standard formatting
    try:
        return date.fromisoformat(iso_str)
    except: 
        return np.nan


# To save datasets
def save_dataset(df, file_name):
    """
    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame to be saved
    file_name : string
        name of the file to save the data in

    """
    # Save the DataFrame into a pickle object
    with open("../data/processed/" + file_name, "wb") as f:
        pickle.dump(df, f)

    # Save to .csv file
    df.to_csv("../data/processed/" + file_name + ".csv", index = False)


# Rename dataset columns
def rename_cols(data, date, content) -> pd.DataFrame: 
    """
    Parameters
    ----------
    data : pd.DataFrame
        pd.DataFrame to be renamed
    date : string
        current name of the column containing DATE information
    content : string
        current name of the column containing CONTENT information
    """
    return data.rename(columns = {date: "date", content: "content"})


################################ Aylien-Exclusive Functions ################################

# To process SOURCE
def get_source(src) -> str:
    """
    Parameters
    ----------
    src : list
        an entry in the "source" column of the Aylien dataset
        
    """
    try:
        return src["home_page_url"]
    except TypeError:
        return np.nan


# To label Aylien
def label(aylien) -> pd.DataFrame:
    """
    Parameters
    ----------
    aylien : pd.DataFrame
        the aylien dataset
    """

    # Get all source names
    all_sources = set(aylien["source"])

    # Load the list of reliable sources we manually compiled.
    rel_src = open("../tools/aylien_reliable_sources.txt", "r")
    rel_src_list = rel_src.read().split("\n")
    rel_src.close()

    # Label the data as TRUE if its source is in the list.
    # Creating the label column
    aylien = pd.concat([aylien, pd.DataFrame(columns = ["reliability"])])

    # Label news from these sources as TRUE.
    for src in rel_src_list:
        aylien.loc[aylien["source"] == src, "reliability"] = 1
    
    # Remove the source column since it will not be a feature available to the model.
    aylien = aylien.drop("source", 1)

    return aylien

################################ FNIR-Exclusive Functions ################################
    """
    Parameters
    ----------
    fnir : pd.DataFrame
        the FNIR dataset
    """
# Remove common gibberish characters present in FNIR
def remove_bad_chars(entry):
    return entry.replace("Â", "")
