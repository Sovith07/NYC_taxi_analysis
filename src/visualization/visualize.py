import pandas as pd
import pathlib
import sys
import joblib
import yaml
from sklearn import metrics
from dvclive import Live

def evaluate(model, X, y, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.
 
     Args:
        live          (dvclive.Live): DVCLive instance.
        model         (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names.
    """
    predictions = model.predict(X)
 
    # Use dvclive to log a few simple metrics...
    mae  = metrics.mean_absolute_error(y, predictions)
    r2   = metrics.r2_score(y, predictions)
 
    if not live.summary:
     live.summary = {"mae": {},  "r2": {}}
     live.summary["mae"][split]  = mae
     live.summary["r2"][split]   = r2
    residuals = y - predictions

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(predictions, residuals, alpha=0.3, s=4)
    plt.axhline(0, color="red", linewidth=0.8)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Residual plot — {split}")
    plt.savefig(f"{save_path}/residuals_{split}.png")
    plt.close()

def save_importance_plot(live, model, feature_names):
       """
       Save feature importance plot.
 
       Args:
       live          (dvclive.Live): DVCLive instance.
       model         (sklearn.ensemble.RandomForestRegressor): Trained Regressor.
       feature_names (list): List of feature names.
    """
       import matplotlib.pyplot as plt
       fig, axes = plt.subplots(dpi=100)
       fig.subplots_adjust(bottom=0.2, top=0.95)
       axes.set_ylabel("Mean decrease in impurity")
 
       importances    = model.feature_importances_
       forest_importances = pd.Series(importances, index=feature_names).nlargest(20)
       forest_importances.plot.bar(ax=axes)
 
       live.log_image("importance.png", fig)

def main():
       
       curr_dir    = pathlib.Path(__file__)
       home_dir    = curr_dir.parent.parent.parent
       model_file=sys.argv[1]
       model=joblib.load(model_file)
       input_file=sys.argv[2]
       data_path=home_dir.as_posix() + input_file

       output_path=home_dir.as_posix() + '/dvclive'
       pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)

       train_features=pd.read_csv(data_path + '/train_csv')
       x_train=train_features.drop(columns=['trip_duration',
        'log_trip_duration','id',
        'pickup_datetime',
        'dropoff_datetime'])
       x_train['store_and_fwd_flag']=x_train['store_and_fwd_flag'].map({
    'N': 0,
    'Y': 1})
       y_train=train_features['log_trip_duration']

       feature_names = x_train.columns.tolist()
       
       test_features=pd.read_csv(data_path + '/test_csv')
       x_test=test_features.drop(columns=['trip_duration',
        'log_trip_duration','id',
        'pickup_datetime',
        'dropoff_datetime'])
       x_test['store_and_fwd_flag']=x_test['store_and_fwd_flag'].map({
    'N': 0,
    'Y': 1})
       y_test=test_features['log_trip_duration']

       with Live(output_path, dvcyaml=False) as live:
        evaluate(model, x_train, y_train, "train", live, output_path)
        evaluate(model, x_test,  y_test,  "test",  live, output_path)
 
        # Dump feature importance plot.
        save_importance_plot(live, model, feature_names)

if __name__ == "__main__":
    main()