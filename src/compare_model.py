from train_randomforest import *
from train_xgboost_pipeline import *

rf_prob = run_randomforest()
xgb_prob = run_xgboost_pipeline()

if xgb_prob > rf_prob:
    print(f"\nXGBoost is more accurate than Random Forest by {xgb_prob - rf_prob:.2f}%.\n")
else:
    print(f"\nRandom Forest is more accurate than XGBoost by {rf_prob - xgb_prob:.2f}%.\n")