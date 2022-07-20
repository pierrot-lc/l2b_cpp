"""Do the plots.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.plots.preprocess_results import read_experience, read_all_results


def save_stats(df: pd.DataFrame, metrics: list, name: str):
    stats = dict()
    for m in metrics:
        stats[m] = [df[m].mean()]
    stats = pd.DataFrame(stats)
    stats.to_csv(name + '.csv', index=False, sep=',')


def mean_by_inst(df: pd.DataFrame, metric: str) -> pd.Series:
    """Retourne la moyenne de la métrique pour chaque instance.
    """
    return df.groupby('instance').mean()[metric]


def mean_by_window(df: pd.DataFrame, metric: str) -> pd.Series:
    """Retourne la moyenne de la métrique pour chaque fenêtre.
    """
    return df.groupby('window').mean()[metric]


def compare_means(
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        name_1: str,
        name_2: str,
        metric: str,
        by_instance: bool,
    ) -> pd.DataFrame:
    mean_fn = mean_by_inst if by_instance else mean_by_window

    serie_1 = mean_fn(df_1, metric)
    serie_2 = mean_fn(df_2, metric)

    serie_1 = serie_1.rename(name_1)
    serie_2 = serie_2.rename(name_2)

    return pd.concat([serie_1, serie_2], axis=1)


def compare_metrics_by_means(
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        name_1: str,
        name_2: str,
        metrics: list,
        by_instance: bool,
    ) -> plt.Figure:
    n_rows = int(np.floor(np.sqrt(len(metrics))))
    n_cols = int(np.ceil(len(metrics) / n_rows))

    fig = plt.figure(figsize=(18, 12))
    axes = fig.subplots(n_rows, n_cols).flatten()
    x_label = 'Instances' if by_instance else 'Windows'

    for metric, ax in zip(metrics, axes):
        df = compare_means(df_1, df_2, name_1, name_2, metric, by_instance)
        df = df.sort_index()
        df.plot.bar(ax=ax)
        ax.set_title(metric)
        ax.set_xlabel('')

    fig.suptitle(f'Mean of metrics, by {x_label.lower()}')

    return fig


def compare_metrics(experiment_dir: str):
    """Compare the metrics between the gencol and the training metrics.
    Save the plots into the experiment folder.
    """
    dfs = read_experience(experiment_dir)

    metrics = [
        'top-1 acc',
        'good but not best',
        'wrong_predictions',
        'nDCG',
    ]


    cfix_metrics = {
        'Frac - ' + m: m
        for m in metrics
    }
    cfix_metrics['instance'] = 'instance'
    cfix_metrics['window'] = 'window'
    df_cfix = dfs['gencol'][cfix_metrics.keys()].copy()
    df_cfix = df_cfix.rename(cfix_metrics, axis=1)

    for plot_name, by_instance in [('instances', True), ('windows', False)]:
        fig = compare_metrics_by_means(
            dfs['train'],
            dfs['gencol'],
            'Train',
            'Gencol',
            metrics,
            by_instance,
        )
        plot_path = os.path.join(experiment_dir, plot_name) + '.png'
        fig.savefig(plot_path)

        fig = compare_metrics_by_means(
            dfs['gencol'],
            df_cfix,
            'iafix',
            'cfix',
            metrics,
            by_instance,
        )
        plot_path = os.path.join(experiment_dir, 'cfix_' + plot_name) + '.png'
        fig.savefig(plot_path)

    metrics = [
        'top-1 acc',
        'approx acc',
        'good but not best',
        'wrong_predictions',
        'nDCG'
    ]
    for df, name in [(dfs['train'], 'stats_train'), (dfs['gencol'], 'stats_gencol')]:
        name = os.path.join(experiment_dir, name)
        save_stats(df, metrics, name)


def compare_solutions(results_dir: str):
    """Compare the solutions of each experiments between each other.
    """
    results = read_all_results(results_dir)

    n_instances = max(
        len(set(dfs['solutions']['instance'].values))
        for dfs in results.values()
    )

    for instance_id in range(1, n_instances+1):
        instance_name = f'instance_{instance_id}'

        series_values, series_nodes = [], []
        for experience_name, dfs in results.items():
            df = dfs['solutions']
            df = df[ df['instance'] == instance_name ]

            sol_values = mean_by_window(df, 'solution')
            sol_values = sol_values.rename(experience_name)

            sol_nodes = mean_by_window(df, 'nodes')
            sol_nodes = sol_nodes.rename(experience_name)

            series_values.append(sol_values)
            series_nodes.append(sol_nodes)

        for series, series_name, series_type in zip(
                [series_values, series_nodes],
                ['Solutions', 'Nodes explored'],
                ['solutions', 'nodes'],
            ):
            df = pd.concat(series, axis=1)
            cfix_1 = df['cfix_1'].values.copy()
            for col in df.columns:
                df[col] = 100 * (df[col] - cfix_1) / cfix_1

            fig = plt.figure(figsize=(18, 12))
            ax = fig.subplots(1, 1)
            df.plot.bar(ax=ax)
            ax.set_title(f'{series_name} of {instance_name}, w.r.t. cfix_1')
            ax.set_xlabel('Windows')
            ax.set_ylabel('%')

            fig_path = os.path.join(results_dir, f'{series_type}_{instance_name}.png')
            fig.savefig(fig_path)
