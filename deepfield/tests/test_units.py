"""Testing module."""
import os
import pathlib
import warnings
import pytest
import numpy as np
import pandas as pd

from ..field import Field

from .data.test_wells import TEST_WELLS

@pytest.fixture(scope="module")
def tnav_model():
    """Load tNav test model."""
    test_path = os.path.dirname(os.path.realpath(__file__))
    tnav_path = os.path.join(test_path, 'data', 'tNav_test_model', 'TEST_MODEL.data')
    return Field(tnav_path, loglevel='ERROR').load()

@pytest.fixture(params=['tnav_model'])
def model(request):
    """Returns model."""
    return request.getfixturevalue(request.param)

#pylint: disable=redefined-outer-name
class TestModelLoad:
    """Testing model load in tNav and HDF5 formats."""
    def test_content(self, model):
        """Testing components and attributes content."""
        assert set(model.components).issubset({'grid', 'rock', 'states', 'tables', 'wells', 'faults'})
        assert set(model.grid.attributes) == {'DIMENS', 'ZCORN', 'COORD', 'ACTNUM'}
        assert set(model.rock.attributes) == {'PORO', }
        assert set(model.states.attributes) == {'PRESSURE', }
        assert set(model.wells.attributes) == {'WELLTRACK', 'WELSPECS'}
        assert len(model.wells.names) == len(TEST_WELLS)

    def test_shape(self, model):
        """Testing data shape."""
        dimens = (2, 1, 6)
        assert np.all(model.grid.dimens.values.ravel() == dimens)
        assert np.all(model.rock.poro.shape == dimens)
        assert np.all(model.grid.actnum.shape == dimens)
        assert model.grid.zcorn.shape == dimens + (8, )
        assert np.all(model.grid.coord.shape == np.array([dimens[0] + 1, dimens[1] + 1, 6]))
        assert model.rock.poro.shape == dimens
        assert model.states.pressure.shape[1:] == dimens

    def test_blocks(self, model):
        """Testing wellblocks."""
        model.wells.get_blocks()
        for i, test in enumerate(TEST_WELLS):
            print(i)
            well = model.wells[str(i)]
            assert np.all(well.welltrack[['X', 'Y', 'Z', 'MD']].values == np.array(test['welltrack']))
            if test['blocks']:
                assert np.all(well.blocks == np.array(test['blocks']))
            else:
                assert well.blocks.size == 0


class TestBenchmarksLoading():
    """Test loading benchmarks. To assighn a path to benchmarks use option --path_to_benchmarks"""
    def test_benchmarks(self, path_to_benchmarks):
        """Test loading models from benchmarks."""

        traverse = pathlib.Path(path_to_benchmarks)
        models_pathways_data_uppercase = list(map(str, list(traverse.rglob("*.DATA"))))
        models_pathways_data_lowercase = list(map(str, list(traverse.rglob("*.data"))))
        models_pathways = models_pathways_data_uppercase + models_pathways_data_lowercase
        if len(models_pathways) > 0:
            failed = []

            for model in models_pathways:
                try:
                    Field(model, loglevel='ERROR').load()
                except Exception as err: #pylint: disable=broad-exception-caught
                    failed.append((model, str(err)))

            errors_df = pd.DataFrame(failed, columns=['Path', 'Error'])
            errors_grouped = []

            for err, df in errors_df.groupby("Error"):
                for record in df.values:
                    errors_grouped.append((err, record[0]))

            errors_grouped_df = pd.DataFrame(errors_grouped, columns=['Error', 'Path'])
            errors_grouped_df.to_csv('errors_grouped.csv', index=False)
            assert len(failed) == 0
        else:
            warnings.warn("Benchmarks folder does not exist")
