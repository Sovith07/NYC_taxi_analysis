import pathlib
import sys
import yaml
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model(train_features, target, n_estimators, max_depth, seed):
    # Train your machine learning model
    train_features['store_and_fwd_flag'] = train_features['store_and_fwd_flag'].map({
    'N': 0,
    'Y': 1
})  
    train_features = train_features.drop(
    columns=[
        'id',
        'pickup_datetime',
        'dropoff_datetime',
    ])
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed
    )
    model.fit(train_features, target)
    return model
 
 
def save_model(model, output_path):
    # Save the trained model to the specified output path
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    joblib.dump(model, output_path + '/model.joblib')
 
 
def main():
    curr_dir    = pathlib.Path(__file__)
    home_dir    = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params      = yaml.safe_load(open(params_file))["train_model"]

    input_file=sys.argv[1]
    data_path=home_dir.as_posix() + input_file
    output_path=home_dir.as_posix() + '/models'

    train_features=pd.read_csv(data_path + '/train_csv')
    x = train_features.drop(columns=['trip_duration',
        'log_trip_duration'])
    y=train_features['log_trip_duration']

    trained_model=train_model(x,y,params['n_estimators'],params['max_depth'],params['seed'])
    save_model(trained_model,output_path)

if __name__ == "__main__":
    main()



