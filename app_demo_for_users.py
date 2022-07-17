import pickle
import pandas as pd

# Load vectorizer
with open("modeling/ML_models/vectorizer", "rb") as f:
    vectorizer = pickle.load(f)

# Load model
with open("modeling/ML_models/model_instances/LR", "rb") as f:
    model = pickle.load(f)

# Take user input
input = input("Enter a COVID news title or a short COVID-related claim: ")

# Process user input
ser = pd.Series(input, index = [0])
X = vectorizer.transform(ser).toarray()

# Let model predict
prediction = model.predict(X)

# Give user ouput
if (prediction[0] == 0):
    print("Your entry is likely false information.")
else:
    print("Your entry is likely true information.")