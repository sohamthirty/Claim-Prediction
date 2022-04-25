# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Data/data_smoteen.csv')

# Loading model to compare the results
model = pickle.load(open('Insurance_Claim_model.pickle','rb'))

rrr= pd.read_csv('Data/classifed.csv')  

print(model.predict(rrr))