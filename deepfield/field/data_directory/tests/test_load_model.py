import pathlib
import pickle
import numpy as np
import pandas as pd

from deepfield.field.data_directory.load_utils import load

def test_load():
    egg_model_path = pathlib.Path('open_data/egg/Egg_Model_ECL.DATA')
    data = load(egg_model_path)
    with open(pathlib.Path(__file__).parent / 'data' / 'egg_loaded_data.pickle', 'rb') as f:
        data_expected = pickle.load(f)
    for section in data_expected:
        for r, e in zip(data[section], data_expected[section], strict=True):
            assert r[0] == e[0]
            if not isinstance(e[1], tuple | list):
                expected_res = [e[1]]
                res = [r[1]]
            else:
                expected_res = e[1]
                res = r[1]
            for r, e in zip(res, expected_res):
                if isinstance(e, np.ndarray):
                    np.testing.assert_equal(r, e)
                elif isinstance(e, pd.DataFrame):
                    pd.testing.assert_frame_equal(r, e)
                else:
                    assert  res == expected_res
