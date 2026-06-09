import pathlib
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from hyperopt import hp,fmin,tpe,space_eval,STATUS_OK,Trials
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def find_best_model_with_params(x_train,y_train,x_test,y_test):
    
    hyperparameters={
        "XGBRegressor":{
            "n_estimators": hp.choice("n_estimators", [10,15,20]),
            "max_depth": hp.choice("max_depth", [6,8,10]),
            "learning_rate": hp.uniform("learning_rate", 0.03, 0.3),
        }
    }

    def eval_model(hyperopt_params):
        params=hyperopt_params
        if 'max_depth' in params: params['max_depth']=int(params['max_depth'])
        if 'n_estimators' in params: params['n_estimators']=int(params['n_estimators'])

        model=XGBRegressor(**params)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)

        model_mse=mean_squared_error(y_test,y_pred)
        mlflow.log_metric('MSE',model_mse)
        loss=model_mse
        return {'loss': loss, 'status': STATUS_OK}
    
    space=hyperparameters['XGBRegressor']
    with mlflow.start_run(run_name='XGBRegressor'):
        argmin=fmin(fn=eval_model,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=5,
                    trials=Trials(),
                    verbose=True)
        #  argmin stores indices
        
    run_ids=[]
    with mlflow.start_run(run_name='XGB Final Model') as run:
        run_id=run.info.run_id
        run_name=run.data.tags['mlflow.runName']
        run_ids +=[(run_name,run_id)]

        #configure params
        params=space_eval(space,argmin) #Convert Indices to Real Values
        if 'max_depth' in params: params['max_depth']=int(params['max_depth'])
        if 'n_estimators' in params: params['n_estimators']=int(params['n_estimators'])
        mlflow.log_params(params)

        model=XGBRegressor(**params)
        model.fit(x_train,y_train)
        mlflow.sklearn.log_model(model,'model')


    return model


 
def save_model(model, output_path):
    # Save the trained model to the specified output path
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    joblib.dump(model, output_path + '/model.joblib')
 
 
def main():
    curr_dir    = pathlib.Path(__file__)
    home_dir    = curr_dir.parent.parent.parent

    input_file=sys.argv[1]
    data_path=home_dir.as_posix() + input_file
    output_path=home_dir.as_posix() + '/models'

    train_features=pd.read_csv(data_path + '/train_csv')
    x_train = train_features.drop(columns=['trip_duration',
        'log_trip_duration'])
    y_train =train_features['log_trip_duration']


    test_features=pd.read_csv(data_path + '/test_csv')
    x_test = test_features.drop(columns=['trip_duration',
        'log_trip_duration'])
    y_test =test_features['log_trip_duration']


    trained_model=find_best_model_with_params(x_train,y_train,x_test,y_test)
    save_model(trained_model,output_path)

if __name__ == "__main__":
    main()



