from script import train_childs, train_simple, train_linear, train_residual


if __name__ == '__main__':
    import os

    valid_names = [
        'simple',
        'childs',
        'linear',
        'residual',
    ]

    if len(os.sys.argv) != 2 or os.sys.argv[1] not in valid_names:
        print(f'Usage: {os.sys.argv[0]} [script_name]')
        print('Accepted scripts:', *valid_names)

        os.sys.exit(0)

    name = os.sys.argv[1]

    if name == 'simple':
        train_simple.main()
    elif name == 'childs':
        train_childs.main()
    elif name == 'linear':
        train_linear.main()
    elif name == 'residual':
        train_residual.main()
