import pandas as pd
import pathlib
import sys
import yaml
from sklearn.model_selection import train_test_split

 
def load_data(file_path):
    return pd.read_csv(file_path)

def concatenate_dataframes(df1, df2):
    """
    Concatenate two dataframes row-wise.
    """
    return pd.concat([df1, df2], ignore_index=True)

def split_data(df, test_split,seed):
    train,test=train_test_split(df,test_size=test_split,random_state=seed)
    return train,test

def save_data(train,test,output_path):
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    train.to_csv(output_path + '/train_csv',index=False)
    test.to_csv(output_path + '/test_csv',index=False)

def main():
    curr_dir=pathlib.Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    params_file=home_dir.as_posix() + '/params.yaml'
    params=yaml.safe_load(open(params_file))['make_dataset']

    input_file_1=sys.argv[1]
    data_path_1=home_dir.as_posix() + input_file_1
    input_file_2=sys.argv[2]
    data_path_2=home_dir.as_posix() + input_file_2
    output_path=home_dir.as_posix() +'/data/processed'

    data_1=load_data(data_path_1)
    data_2=load_data(data_path_2)

    complete_df=concatenate_dataframes(data_1,data_2)

    train_data,test_data=split_data(complete_df,params['test_split'],params['seed'])

    save_data(train_data,test_data,output_path)

if __name__ == "__main__":
    main()


