from train_randomforest import *
from train_xgboost_pipeline import *

def compare_model():
    rf_prob, rf_model = run_randomforest()
    xgb_prob, xgb_model = run_xgboost_pipeline()

    if xgb_prob > rf_prob:
        print(f"\nXGBoost is more accurate than Random Forest by {xgb_prob - rf_prob:.2f}%.\n")
        return xgb_model
    else:
        print(f"\nRandom Forest is more accurate than XGBoost by {rf_prob - xgb_prob:.2f}%.\n")
        return rf_model