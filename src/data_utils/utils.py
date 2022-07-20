"""Any util function about the data that does not fit into any other module.
"""

FRACVAL_ID = 4

EUCLIDIAN_COLUMNS = [
    'var_cost',
    'fraction_conflicting_columns',
    'fraction_conflicting_columns_positive_value',
    'min_cost_conflicting_column',
    'min_cost_conflicting_column_positive_value',
    'dual_cost_min',
    'dual_cost_max',
    'dual_cost_avg',
    'nb_pairing_tasks',
]

LOG_COLUMNS = [
    'fraction_conflicting_columns',
    'fraction_conflicting_columns_positive_value',
    'number_cols_in_mp',
    'dual_cost_max',
]


def parse_tree_name(name: str, dagger: bool) -> dict:
    if dagger:
        name = '_'.join(name.split('_')[1:])

    num = name.split('_win_')[-1].split('.')[0]
    instance_name = name[:len('instance_x')]
    params = name[len('instance_x'):].split('_win_')[0]

    return {
        'instance': instance_name,
        'params': params,
        'window': int(num) + 1,
    }
