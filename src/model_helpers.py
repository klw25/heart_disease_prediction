from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

from data import load_data

def load_values():
    data = load_data()

    y = data.HeartDisease
    ignore_cols = ['HeartDisease', 'Age', 'Sex']
    features = [col for col in data if col not in ignore_cols]
    x = data[features]

    x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=1)

    return x_train, x_val, y_train, y_val, x, y

def cat_to_num(x_train, x_val):
    #TODO - need to figure out how many options for each categorical variable
    s = (x_train.dtypes == 'object')
    object_cols = list(s[s].index)

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(x_val[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = x_train.index
    OH_cols_valid.index = x_val.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = x_train.drop(object_cols, axis=1)
    num_X_valid = x_val.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_x_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_x_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    # Ensure all columns have string type
    OH_x_train.columns = OH_x_train.columns.astype(str)
    OH_x_valid.columns = OH_x_valid.columns.astype(str)

    return OH_x_train, OH_x_valid


def get_mae(max_leaf_nodes, x_train, x_val, y_train, y_val):
    OH_x_train, OH_x_val = cat_to_num(x_train, x_val)

    decisiontree_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    decisiontree_model.fit(OH_x_train, y_train)
    val_predictions = decisiontree_model.predict(OH_x_val)
    mae = mean_absolute_error(y_val, val_predictions)
    return mae