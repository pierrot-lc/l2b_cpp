"""Unzip the files and preprocess the data.
"""
import os
import shutil

import pandas as pd


USELESS_COLUMNS = [
    'number_conflicting_columns',
    'number_conflicting_columns_positive_value',
    'nb_fractional_vars',
    'nb_tasks',
]


def rename(instance_name: str) -> str:
    """Rename the instances to have them ordered by size.
    """
    if 'NW_727' in instance_name:
        instance_id = 1
    elif 'NW_DC9' in instance_name:
        instance_id = 2
    elif 'NW_D94' in instance_name:
        instance_id = 3
    elif 'NW_D95' in instance_name:
        instance_id = 4
    elif 'NW_757' in instance_name:
        instance_id = 5
    elif 'NW_319' in instance_name:
        instance_id = 6
    elif 'NW_320' in instance_name:
        instance_id = 7
    else:
        print(f'Error: instance {instance_name} not recognized!')
        return instance_name

    instance_name = f'instance_{instance_id}_{instance_name.split("_test_")[1]}'
    return instance_name


def rename_instances(dir_instances: str):
    """Rename all instances in the directory.
    """
    for f in os.listdir(dir_instances):
        filepath = os.path.join(dir_instances, f)
        n_f = rename(f)
        n_filepath = os.path.join(dir_instances, n_f)
        shutil.move(filepath, n_filepath)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataframe.

    Actually, it only removes some useless columns.
    The rest is done in real time during training.
    """
    return df.drop(columns=USELESS_COLUMNS)


def normalize_instances(dir_instances: dir):
    """Normalize all the instances in the directory.
    Also when saving the df, it modifies the separation
    character with ','.
    """
    for f in os.listdir(dir_instances):
        filepath = os.path.join(dir_instances, f)
        df = pd.read_csv(filepath, sep='\t')
        df = normalize_df(df)
        df.to_csv(filepath, index=False, sep=',')


def filter_instances(dir_instances: str):
    """Remove unwanted instances.

    Those are the instances of the 8th window (i.e. with `_win_7` in its name).
    Also remove empty files.
    """
    to_remove = []
    for f in os.listdir(dir_instances):
        filepath = os.path.join(dir_instances, f)
        df = pd.read_csv(filepath, sep='\t')

        if df.empty or '_win_7' in f:
            to_remove.append(filepath)

    for filepath in to_remove:
        os.remove(filepath)


def preprocess(dir_data: str):
    """Filter, normalize and rename all the data.
    """
    print('Preprocessing...')
    train_dir = os.path.join(dir_data, 'train')
    test_dir = os.path.join(dir_data, 'test')

    for dir_instances in [train_dir, test_dir]:
        filter_instances(dir_instances)
        rename_instances(dir_instances)
        normalize_instances(dir_instances)
