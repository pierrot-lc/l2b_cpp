"""Read and preprocess the metrics and the scores.
"""
import os

import pandas as pd
import numpy as np


def rename_gencol_instances(df: pd.DataFrame):
    """Only keeps the basename of each instances.
    """
    modify_name = lambda inst: inst[:-len('_legX_credX')]
    df['instance'] = df['instance'].apply(modify_name)


def read_experience(dir_path: str) -> dict:
    if os.path.exists(os.path.join(dir_path, 'metrics_gencol.csv')):
        df_gencol = pd.read_csv(os.path.join(dir_path, 'metrics_gencol.csv'))
        rename_gencol_instances(df_gencol)
    else:
        df_gencol = None

    if os.path.exists(os.path.join(dir_path, 'metrics_training.csv')):
        df_train = pd.read_csv(os.path.join(dir_path, 'metrics_training.csv'))
    else:
        df_train = None

    df_sol = pd.read_csv(os.path.join(dir_path, 'solution.csv'))

    rename_gencol_instances(df_sol)

    return {
        'gencol': df_gencol,
        'train': df_train,
        'solutions': df_sol,
    }

def read_all_results(results_path: str) -> dict:
    experiment_dirs = [
        os.path.join(results_path, exp_dir)
        for exp_dir in os.listdir(results_path)
        if os.path.isdir(os.path.join(results_path, exp_dir)) and exp_dir != 'save_old_results'
    ]

    results = dict()

    for dir_path in experiment_dirs:
        if dir_path.endswith('/'):
            dir_path = dir_path[:-1]  # Remove pending '/' to get a proper basename

        experience_name = os.path.basename(dir_path)
        results[experience_name] = read_experience(dir_path)

    return results
