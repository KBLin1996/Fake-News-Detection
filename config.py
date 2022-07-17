"""
    Environmental variables
    -----------------------
    "save_pickle": bool
        if True, save the dataset as a pickle object
    "save_csv": bool
        if True, save the dataset as a csv file
    "seed": int
        integer for the seed value; fixed seed for reproducibility
"""

config = {
    "save_pickle": True,
    "save_csv": True,
    "save_model": True,
    "use_clean": False,
    "use_date": False,
    "save_misclassified": True,
    "seed": 42
}