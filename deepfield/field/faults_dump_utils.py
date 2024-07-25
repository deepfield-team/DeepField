"""Dump tools."""
import pandas as pd

def write_faults(f, faults):
    """Write FAULTS to file."""
    dfs = []
    for node in faults:
        if 'FAULTS' in node.attributes and not node.faults.empty:
            faults = node.faults.copy()
            dfs.append(faults)
    if not dfs:
        return

    df = pd.concat(dfs, sort=False).sort_values('NAME')
    df['END_LINE'] = '/'

    f.write('FAULTS\n')
    f.write(df.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('/\n\n')

def write_multflt(f, faults):
    """Write MULTFLT to file."""
    dfs = []
    for node in faults:
        if 'MULTFLT' in node.attributes and not node.multflt.empty:
            multflt = node.multflt.copy()
            dfs.append(multflt)
    if not dfs:
        return

    df = pd.concat(dfs, sort=False).sort_values('NAME')
    df['END_LINE'] = '/'

    f.write('MULTFLT\n')
    f.write(df.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('/\n\n')