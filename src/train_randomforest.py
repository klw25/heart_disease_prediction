from data import load_data
from model_helpers import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

import pandas as pd
import matplotlib.pyplot as plt

def run_randomforest():
    x_train, x_val, y_train, y_val, x, y = load_values()

    max_leaf_nodes = [5, 25, 50, 100, 250, 500, 600, 700, 1000, 10000]

    mae = 1000000000000
    ideal = 0
    for max_leaf_nodes in max_leaf_nodes:
        if mae > get_mae(max_leaf_nodes, x_train, x_val, y_train, y_val):
            mae = get_mae(max_leaf_nodes, x_train, x_val, y_train, y_val)
            ideal = max_leaf_nodes
    print(f"The ideal max leaf nodes for the is {ideal}.")

    OH_x_train, OH_x_val = cat_to_num(x_train, x_val)


    randomforest_model = RandomForestClassifier(max_leaf_nodes=ideal, random_state=1)
    randomforest_model.fit(OH_x_train, y_train)
    val_predictions = randomforest_model.predict(OH_x_val)

    score = accuracy_score(y_val, val_predictions)

    print(f"The rf model is {score * 100:.2f}% accurate.")

    #must do with pipeline
    '''scores = cross_val_score(randomforest_model, x, y,
                         cv=5,
                         scoring='roc_auc')
    
    print(f"Average AUC Score (with cross validation scores):", scores.mean())
    print("\n")'''

    return score * 100
