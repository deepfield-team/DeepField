"""Miscellaneous utils."""
from pathlib import Path
import subprocess
import signal
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
