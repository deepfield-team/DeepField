"""Miscellaneous utils."""
import glob
import os
from pathlib import Path
import re
import subprocess
import signal
import fnmatch
from contextlib import contextmanager
import numpy as np
import psutil
from tqdm import tqdm


@contextmanager
def _dummy_with():
    """Dummy statement."""
    yield

def kill(proc_pid):
    """Kill proc and its childs."""
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def signal_handler(signum, frame):
    """Timeout handler."""
    _ = signum, frame
    raise TimeoutError("Timed out!")

def execute_tnav_models(models, license_url,
                        tnav_path, base_script_path=None, logfile=None,
                        global_timeout=None, process_timeout=None,
                        dump_rsm=True, dump_egrid=True, dump_unsmry=True, dump_unrst=True):
    """Execute a bash script for each model in a set of models.

    Parameters
    ----------
    models : str, list of str
        A path to model or list of pathes.
    license_url : str
        A license server url.
    tnav_path : str
        A path to tNavigator executable.
    base_script_path : str
        Path to script to execute.
    logfile : str
        A path to file where to point stdout and stderr.
    global_timeout : int
        Global timeout in seconds.
    process_timeout : int
        Process timeout. Kill process that exceeds the timeout and go to the next model.
    dump_rsm: bool
        Dump *.RSM file, by default True.
    dump_egrid: bool
        Dump *.EGRID file, by default False.
    dump_unsmry: bool
        Dump *.SMSPEC and *.UNSMRY files, by default False.
    dump_unrst: bool
        Dump *.UNRST file, by default True.
    """
    if base_script_path is None:
        base_script_path = Path(__file__).parents[2] / 'bin/tnav_run.sh'
    if license_url is None:
        raise ValueError('License url is not defined.')
    models = np.atleast_1d(models)
    keys = ''
    if dump_egrid:
        keys += 'e'
    if dump_unrst:
        keys += 'r'
    if dump_unsmry:
        keys += 'um'
    if len(keys) > 0:
        keys = '-' + keys
    if dump_rsm:
        keys += ' --ecl-rsm'

    base_args = ['bash', base_script_path, tnav_path, license_url, keys,]
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(-1 if global_timeout is None else global_timeout)
    with (open(logfile, 'w') if logfile is not None else _dummy_with()) as f:#pylint:disable=consider-using-with
        for model in tqdm(models):
            try:
                p = subprocess.Popen(base_args + [model], stdout=f, stderr=f)#pylint:disable=consider-using-with
                try:
                    p.wait(timeout=process_timeout)
                except subprocess.TimeoutExpired:
                    kill(p.pid)
            except Exception as err:
                kill(p.pid)
                raise err

def recursive_insensitive_glob(path, pattern, return_relative=False):
    """Find files matching pattern ignoring case-style."""
    found = []
    reg_expr = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    for root, _, files in os.walk(path, topdown=True):
        for f in files:
            if re.match(reg_expr, f):
                f_path = os.path.join(root, f)
                found.append(
                    f_path if not return_relative else os.path.relpath(f_path, start=path)
                )
    return found

def get_single_path(dir_path, filename, logger=None):
    """Find a file withihn the directory. Raise error if multiple files found."""
    files = recursive_insensitive_glob(dir_path, filename)
    if not files:
        if logger is not None:
            logger.warning("{} file was not found.".format(filename))
        return None
    if len(files) > 1:
        raise ValueError('Directory {} contains multiple {} files.'.format(dir_path, filename))
    return files[0]

def rolling_window(a, strides):
    """Rolling window without overlays."""
    strides = np.asarray(strides)
    output_shape = tuple((np.array(a.shape) - strides)//strides + 1) + tuple(strides)
    output_strides = tuple(strides * np.asarray(a.strides)) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=output_shape, strides=output_strides)

def get_multout_paths(path_to_results, basename, extension_regexp=r'X\d+'):
    r"""Searches for multout files in a RESULT folder near .DATA file.

    Parameters
    ----------
    path_to_results: str
        Path to the folder with precomputed results of hydrodynamical simulation.
    basename: str
        Model Name.
    extension_regexp: str, optional
        Regexp to match a file extension, by default r'X\d+'.

    Returns
    -------
    paths: list or None
        List of the paths found. None else.
    """
    multout_paths = []
    for filename in glob.iglob(os.path.join(path_to_results, '**', basename+'.*'), recursive=True):
        _, ext = os.path.splitext(filename)
        if re.fullmatch(extension_regexp, ext[1:]):
            multout_paths.append(filename)
    return sorted(multout_paths) if multout_paths else None
