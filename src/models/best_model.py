from lazypredict.Supervised import LazyRegressor
import pathlib
import sys
import pandas as pd

# Load dataset
def best_model(train_features, train_target, test_features, test_target):


 offset = int(train_features.shape[0] * 0.9)

  # Split data into training and testing sets
 X_train, y_train = train_features[:offset], train_target[:offset]
 X_test, y_test = test_features[offset:], test_target[offset:]

 # Initialize LazyRegressor and fit data
 reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
 models, predictions = reg.fit(X_train, X_test, y_train, y_test)
 
 print(models)
 get_best_model = models["R-Squared"].idxmax()
 return get_best_model


def main():
    curr_dir    = pathlib.Path(__file__)
    home_dir    = curr_dir.parent.parent.parent

    input_file=sys.argv[1]
    data_path=home_dir.as_posix() + input_file

    train_features=pd.read_csv(data_path + '/train_csv')
    x_train = train_features.drop(columns=['trip_duration',
        'log_trip_duration'])
    y_train=train_features['log_trip_duration']

    test_features=pd.read_csv(data_path + '/test_csv')
    x_test= train_features.drop(columns=['trip_duration',
        'log_trip_duration'])
    y_test=train_features['log_trip_duration']

    get_best_model=best_model(x_train,y_train,x_test,y_test)
    print(f"Best Model: {get_best_model}")
    #save_model(trained_model,output_path)

if __name__ == "__main__":
    main()
