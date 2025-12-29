from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from model_helpers import *

from xgboost import XGBClassifier

def run_xgboost_pipeline():
    x_train, x_val, y_train, y_val, x, y = load_values()

    s = (x_train.dtypes == 'object')
    object_cols = list(s[s].index)

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, object_cols)
        ]
    )

    xgboost_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)


    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', xgboost_model)])
    
    my_pipeline.fit(x_train, y_train,  
                    model__verbose=False)
    
    val_predictions = my_pipeline.predict(x_val)

    score = accuracy_score(y_val, val_predictions)

    print(f"The xgboost pipeline model is {score * 100:.2f}% accurate.")

    scores = cross_val_score(my_pipeline, x, y,
                         cv=5,
                         scoring='roc_auc')
    
    print("Average AUC Score (with cross validation scores):", scores.mean())

    return score * 100, my_pipeline
