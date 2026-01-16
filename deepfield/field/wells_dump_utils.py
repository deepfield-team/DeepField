"""Dump tools."""
import pandas as pd

PERF_VALUE_COLUMNS = ['RAD', 'DIAM', 'SKIN', 'MULT']

def write_perf(f, wells, defaults):
    """Write perforations to file."""
    dfs = []
    for node in wells:
        if 'PERF' in node.attributes:
            if node.perf.empty:
                continue
            df = node.perf.copy()
            df['WELL'] = node.name
            dfs.append(df)
    if dfs:
        df = pd.concat(dfs, sort=False)
    else:
        return

    f.write('EFORm WELL \'DD.MM.YYYY\' MDL MDU ' +
            ' '.join([c for c in df if c in PERF_VALUE_COLUMNS]) + '\n')
    f.write('ETAB\n')

    for c in df.columns:
        if c in defaults:
            df[c] = df[c].fillna(defaults[c])
    if df.isna().any().any():
        raise ValueError('Perforations contains nans.')

    df['PERF'] = 'PERF'
    df['BRANCH'] = df['WELL'].str.split(':').apply(lambda x: 'BRANCH {}'.format(':'.join(x[1:]))
                                                   if len(x) > 1 else '')
    df['WELL'] = df['WELL'].str.split(':').apply(lambda x: x[0])
    df['DATE'] = df['DATE'].dt.strftime('%d.%m.%Y')
    if 'CLOSE' in df:
        df['CLOSE'] = df['CLOSE'].apply(lambda x: 'CLOSE' if x else '')

    df = df[['WELL', 'DATE', 'PERF', 'MDL', 'MDU'] +
            [c for c in df if c in PERF_VALUE_COLUMNS] +
            ['BRANCH'] +
            (['CLOSE'] if 'CLOSE' in df else [])]

    f.write(df.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('ENDE\n')

def expand_event_df(df, value_control_kw):
    """Add control keywords to columns."""
    order = []
    for col in df.columns:
        if col in value_control_kw:
            df[col + '_'] = col
            order.extend([col + '_', col])
        elif col == 'MODE':
            order = ['MODE'] + order
    order = ['WELL', 'DATE'] + order
    return df[order]

def write_events(f, wells, value_control_kw):
    """Write perforations to file."""
    dfs = []
    for node in wells:
        if 'EVENTS' in node.attributes:
            if node.events.empty:
                continue
            df = node.events.copy()
            df['WELL'] = node.name
            dfs.append(df)
    if dfs:
        df = pd.concat(dfs, sort=False)
    else:
        return

    f.write('EFORm WELL \'DD.MM.YYYY\'\n')
    f.write('ETAB\n')

    df = df[['WELL'] + [col for col in df if col != 'WELL']]

    df['DATE'] = df['DATE'].dt.strftime('%d.%m.%Y')

    if 'MODE' in df:
        for _, df_mode in df.groupby('MODE'):
            df_mode = df_mode.dropna(axis=1)
            df_mode = expand_event_df(df_mode, value_control_kw)
            f.write(df_mode.to_string(header=False, index=False, index_names=False) + '\n')
    else:
        df = df.dropna(axis=1)
        f.write(df_mode.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('ENDE\n')

