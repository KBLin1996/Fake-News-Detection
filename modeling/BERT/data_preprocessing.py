import pandas as pd

def ReadCSV(include_date=False):
    # Sperate the text and the label
    df = pd.read_csv('final.csv')
    texts = list()
    labels = list()

    for _, row in df.iterrows():
        if include_date:
            texts.append(f"{row[0][5:]}, {row[1]}")
        else:
            texts.append(row[1])
        labels.append(row[2])

    return texts, labels