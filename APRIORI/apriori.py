# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 23:25:51 2018

@author: Pavel
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mall dataset
dataset = pd.read_csv('data/Market_Basket_Optimisation.csv',
                      header=None)

# Iterating through the dataset to create a list of lists 
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(len(dataset.iloc[0, :]))])
    
# Training Apriori on the dataset
import sys
sys.path.append('data')
from apyori import apriori
rules = apriori(transactions, min_support=3*7/len(dataset), min_confidence=0.2,
                min_lift=3, min_length=2)

# Visualising the results
results = list(rules)