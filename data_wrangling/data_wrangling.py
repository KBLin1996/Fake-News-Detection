"""
    This file contains code for the data wrangling procedures and can save intermediate results according to configurations.
"""

import json
import pickle
import random
import numpy as np
import pandas as pd

from datetime import date, datetime

# Custom imports
import sys
sys.path.append("../")

from config import config
from utils import *



def preprocess_aylien():
    """
        Take a subset of the original Aylien dataset and save it to a pickle object for future use.
    """
    # Dealing with JSONL dataset file
    # BEGIN: CREDIT GOES TO https://galea.medium.com/how-to-love-jsonl-using-json-line-format-in-your-workflow-b6884f65175b

    sample_size = 50000
    aylien_list = []
    random.seed(config["seed"])

    with open("../data/raw/aylien_covid_news_data.jsonl", 'r', encoding='utf-8') as f:   
        total_num_lines = sum(1 for line in f)
        indices = random.sample(range(total_num_lines), sample_size)
        
        f.seek(0)
        
        for i, line in enumerate(f):
            if i in indices:
                aylien_list.append(json.loads(line.rstrip('\n|\r')))

    # END: CREDIT GOES TO https://galea.medium.com/how-to-love-jsonl-using-json-line-format-in-your-workflow-b6884f65175b

    with open("../data/raw/aylien_preprocessed", "wb") as aylien_preprocessed_file:
        pickle.dump(aylien_list, aylien_preprocessed_file)



def prep_aylien():
    """
        Clean up and prepare the Aylien dataset.
    """
    
    # Preprocess and create pickle object to use
    preprocess_aylien()

    # Open file
    with open("../data/raw/aylien_preprocessed", "rb") as aylien_preprocessed_file:
        aylien = pd.DataFrame(pickle.load(aylien_preprocessed_file))


    # Drop entries that have null values
    aylien = aylien.dropna()


    # Drop unnecessary columns first
    aylien = aylien.drop(['author', 'body','categories', 'characters_count', 'entities', 'hashtags', 'id', 'keywords', 'language', 
                    'links', 'media','paragraphs_count','sentences_count', 'sentiment', 'social_shares_count', 'summary', 'words_count'], 1)  


    # Rename Columns for Consistency
    aylien = aylien.rename(columns = {"published_at": "date", "title": "content"})


    # Process DATE
    aylien["date"] = aylien["date"].apply(get_date)


    # Process SOURCE
    aylien["source"] = aylien["source"].apply(get_source)


    # Previous operations might have added some more NA values
    aylien = aylien.dropna()


    # Label the Aylien dataset
    aylien = label(aylien)


    # Split labeled / unlabeled parts
    aylien_true = aylien[aylien["reliability"] == 1]
    aylien_unlabeled = aylien[aylien["reliability"] != 1]


    # Save the labeled / unlabeled parts separately if we specified to save it
    if config["save_pickle"]:
        with open("../data/processed/aylien_true", "wb") as aylien_true_file:
            pickle.dump(aylien_true, aylien_true_file)
        with open("../data/processed/aylien_unlabeled", "wb") as aylien_unlabeled_file:
            pickle.dump(aylien_unlabeled, aylien_unlabeled_file)
    
    if config["save_csv"]:
        aylien_true.to_csv("../data/processed/aylien_true.csv", index = False)
        aylien_unlabeled.to_csv("../data/processed/aylien_unlabeled.csv", index = False)

    return aylien, aylien_true, aylien_unlabeled



def prep_fnir():

    """
        Clean up and prepare the FNIR dataset.
    """

    # Load data
    fnir_fake = pd.read_csv("../data/raw/fakeNews.csv", encoding = "ISO-8859-1")

    # Drop entries that have null values
    fnir_fake = fnir_fake.dropna()

    # Drop unnecessary columns
    fnir_fake = fnir_fake.drop(["Link", "Region", "Country", "Explanation", "Origin", "Origin_URL", "Fact_checked_by", "Poynter_Label"], 1)

    # Rename Columns for Consistency
    fnir_fake = fnir_fake.rename(columns = {"Date Posted": "date", "Text": "content", "Binary Label": "reliability"})
        
    # Process DATE
    fnir_fake["date"] = fnir_fake["date"].apply(get_date)

    # Save Datasets
    if config["save_pickle"]:   
        with open("../data/processed/fnir_fake", "wb") as fnir_fake_file:
            pickle.dump(fnir_fake, fnir_fake_file)

    if config["save_csv"]:
        fnir_fake.to_csv("../data/processed/fnir_fake.csv", index = False)
    
    return fnir_fake



def get_final_data():

    """
        Merge the processed Aylien and FNIR datasets and return it as our final dataset.
    """

    # Combine Datasets
    final = pd.concat([prep_aylien(), prep_fnir()], ignore_index = True)

    if config["save_pickle"]:
        with open("../data/processed/final", "wb") as final_file:
            pickle.dump(final, final_file)

    if config["save_csv"]:
        final.to_csv("../data/processed/final.csv", index = False)
    
    return final
