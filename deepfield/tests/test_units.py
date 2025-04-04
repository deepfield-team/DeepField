"""Testing module."""
import os
import pathlib
import warnings
import pytest
import numpy as np
import pandas as pd

from ..field import Field, OrthogonalGrid
from ..field.base_component import BaseComponent
from ..field.base_spatial import SpatialComponent
from ..field.getting_wellblocks import defining_wellblocks_vtk

from .data.test_wells import TEST_WELLS

@pytest.fixture(scope="module")
def tnav_model():
    """Load tNav test model."""
    test_path = os.path.dirname(os.path.realpath(__file__))
    tnav_path = os.path.join(test_path, 'data', 'tNav_test_model', 'TEST_MODEL.data')
    return Field(tnav_path, loglevel='ERROR').load()

@pytest.fixture(scope="module")
def hdf5_model():
    """Load HDF5 test model."""
    test_path = os.path.dirname(os.path.realpath(__file__))
    hdf5_path = os.path.join(test_path, 'data', 'hdf5_test_model', 'test_model.hdf5')
    return Field(hdf5_path, loglevel='ERROR').load()

@pytest.fixture(scope='module')
def arithmetics_model():
    """Load model with arithmetics"""
    test_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(test_path, 'data', 'arithmetics_test_model', 'test_model.data')
    return Field(model_path, loglevel='ERROR').load()

@pytest.fixture(params=['tnav_model', 'hdf5_model'])
def model(request):
    """Returns model."""
    return request.getfixturevalue(request.param)

#pylint: disable=redefined-outer-name
class TestModelLoad:
    """Testing model load in tNav and HDF5 formats."""
    def test_content(self, model):
        """Testing components and attributes content."""
        assert set(model.components).issubset({'grid', 'rock', 'states', 'tables', 'wells', 'aquifers', 'faults'})
        assert set(model.grid.attributes) == {'DIMENS', 'ZCORN', 'COORD', 'ACTNUM', 'MAPAXES'}
        assert set(model.rock.attributes) == {'PORO', }
        assert set(model.states.attributes) == {'PRESSURE', }
        assert len(model.wells.names) == len(TEST_WELLS)
        assert set(model.meta.keys()) == {'FLUIDS', 'UNITS', 'HUNITS', 'DATES',
                                          'START', 'TITLE', 'MODEL_TYPE', 'SUMMARY'}

    def test_shape(self, model):
        """Testing data shape."""
        dimens = (2, 1, 6)
        assert np.all(model.grid.dimens == dimens)
        assert np.all(model.rock.poro.shape == model.grid.dimens)
        assert np.all(model.grid.actnum.shape == model.grid.dimens)
        assert model.grid.zcorn.shape == dimens + (8, )
        assert np.all(model.grid.coord.shape == np.array([dimens[0] + 1, dimens[1] + 1, 6]))
        assert model.grid.mapaxes.shape == (6, )
        assert model.rock.poro.shape == dimens
        assert model.states.pressure.shape[1:] == dimens

    def test_dtype(self, model):
        """Testing data types."""
        def isfloat_32_or_64(x):
            return np.issubdtype(x.dtype, np.floating)
        assert np.issubdtype(model.grid.dimens.dtype, np.integer)
        assert np.issubdtype(model.grid.actnum.dtype, bool)
        assert isfloat_32_or_64(model.grid.zcorn)
        assert isfloat_32_or_64(model.grid.coord)
        assert isfloat_32_or_64(model.rock.poro)
        assert isfloat_32_or_64(model.states.pressure)


class TestPipeline():
    """Testing methods in pipelines."""
    def test_wells_pipeline(self, hdf5_model): #pylint: disable=redefined-outer-name
        """Testing wells processing."""
        model = hdf5_model.copy()
        model.wells.update({'no_welltrack': {'perf': pd.DataFrame()}})
        model.wells.drop_incomplete()
        assert 'no_welltrack' not in model.wells.names
        model.wells.get_blocks()
        assert np.all(['BLOCKS' in node for node in model.wells])
        model.wells.drop_outside()
        assert len(model.wells.names) == 25
        assert min((node.blocks.size for node in model.wells)) > 0


class TestBaseComponent():
    """Testing BaseComponent."""

    def test_case(self):
        """Testing attrbutes are case insensitive."""
        bcomp = BaseComponent()
        bcomp.sample_attr = 1
        assert bcomp.SAMPLE_ATTR == 1
        assert bcomp.sample_attr == 1
        assert set(bcomp.attributes) == {'SAMPLE_ATTR', }

    def test_read_arrays(self):
        """Testing read arrays."""
        bcomp = BaseComponent()
        bcomp._read_buffer(['0 1 2 1*3\n2*4 5/'], attr='compr', compressed=True, dtype=int) #pylint:disable=protected-access
        bcomp._read_buffer(['0 1 2 3 4 4 5/'], attr='flat', compressed=False, dtype=int) #pylint:disable=protected-access

        assert isinstance(bcomp.flat, np.ndarray)
        assert isinstance(bcomp.compr, np.ndarray)
        assert bcomp.flat.shape == bcomp.compr.shape
        assert bcomp.flat.shape == (7, )
        assert np.all(bcomp.flat == bcomp.compr)

    def test_state(self):
        """Testing state."""
        bcomp = BaseComponent()
        bcomp.init_state(test=True)
        assert bcomp.state.test
        bcomp.set_state(test=False)
        assert ~bcomp.state.test


class TestSpatialComponent():
    """Testing SpatialComponent."""

    def test_ravel(self):
        """Testing ravel state."""
        data = np.arange(10).reshape(2, 5)
        comp = SpatialComponent()
        comp.arr = data
        data_ravel = comp.ravel('ARR')
        assert data_ravel.shape == (10,)
        assert np.all(data_ravel == data.ravel(order='F'))


@pytest.fixture(scope="module")
def orth_grid():
    """Provides orthogonal uniform grid."""
    grid = OrthogonalGrid(dimens=np.array([4, 6, 8]),
                          dx=np.ones([4, 6, 8]),
                          dy=np.ones([4, 6, 8]),
                          dz=np.ones([4, 6, 8]),
                          tops=np.zeros([4, 6, 8]) + np.arange(8),
                          actnum=np.ones((4, 6, 8)))
    return grid

class TestOrthogonalGrid():
    """Testing orthogonal uniform grids."""

    def test_setup(self, orth_grid): #pylint: disable=redefined-outer-name
        """Testing grid setup."""
        assert np.all(orth_grid.dimens == [4, 6, 8])
        assert np.all(orth_grid.cell_volumes == 1)
        assert np.all(np.isclose(orth_grid.xyz, orth_grid.as_corner_point.xyz))
        assert np.all(np.isclose(orth_grid.cell_centroids, orth_grid.as_corner_point.cell_centroids))

    def test_upscale(self, orth_grid): #pylint: disable=redefined-outer-name
        """Testing grid upscale and downscale methods."""
        upscaled = orth_grid.upscale(2)
        assert np.all(upscaled.dimens == orth_grid.dimens / 2)
        assert np.all(upscaled.cell_volumes == 8)
        assert np.all(upscaled.actnum)
        downscaled = orth_grid.downscale(2)
        assert np.all(downscaled.dimens == orth_grid.dimens * 2)
        assert np.all(downscaled.cell_volumes == 1/8)
        assert np.all(downscaled.actnum)


class TestCornerPointGrid():
    """Testing corner-point grids."""

    def test_setup(self, hdf5_model): #pylint: disable=redefined-outer-name
        """Testing grid setup."""
        grid = hdf5_model.grid
        assert np.all(grid.cell_volumes == 1)

    def test_upscale(self, hdf5_model): #pylint: disable=redefined-outer-name
        """Testing grid upscale and downscale methods."""
        grid = hdf5_model.grid
        upscaled = grid.upscale(factors=grid.dimens)
        assert np.all(upscaled.dimens == [1, 1, 1])
        assert np.all(upscaled.coord == [[[0., 0., 0., 0., 0., 6.],
                                          [0., 1., 0., 0., 1., 6.]],
                                         [[2., 0., 0., 2., 0., 6.],
                                          [2., 1., 0., 2., 1., 6.]]])
        assert np.all(upscaled.zcorn == [0., 0., 0., 0., 6., 6., 6., 6.])


class TestWellblocks():
    """Testing algorithm for defining wellblocks. """

    def test_algorithm(self):
        """Creating test wells and check blocks and intersections for every block."""
        grid = OrthogonalGrid(dimens=np.array([2, 1, 6]),
                              dx=np.ones([2, 1, 6]),
                              dy=np.ones([2, 1, 6]),
                              dz=np.ones([2, 1, 6]),
                              actnum=np.ones([2, 1, 6]).astype(bool))
        grid = grid.to_corner_point()

        grid.actnum[0, 0, 1] = False
        grid.actnum[0, 0, 4] = False
        grid.actnum[1, 0, 1] = False
        grid.actnum[1, 0, 4] = False

        grid.create_vtk_locator()

        for test_well in TEST_WELLS:
            output = defining_wellblocks_vtk(test_well['welltrack'], '1', grid,
                                             grid._vtk_locator, grid._cell_id_d) #pylint: disable=protected-access
            xyz_block, _, _, inters = output
            for i, block in enumerate(xyz_block):
                assert np.allclose(
                    test_well['blocks'][i], block), f"Error in defining blocks: {test_well['blocks'], xyz_block}"
                assert np.allclose(
                    test_well['inters'][i], inters[i]), f"Error in defining intersections {test_well['inters'], inters}"

class TestArithmetics():
    """Test loading model with arithmetics keywords."""
    def test_arithmetics(self, arithmetics_model):
        """Test loading model with arithmetics keywords."""
        assert arithmetics_model.rock.permx is not None
        assert arithmetics_model.rock.permy is not None
        assert arithmetics_model.rock.permy is not None
        assert np.allclose(arithmetics_model.rock.permx, arithmetics_model.rock.poro * 500)
        assert np.allclose(arithmetics_model.rock.permz, arithmetics_model.rock.permx * 0.1)
        assert np.allclose(arithmetics_model.rock.permy[3:6, 3:6, 1:1], arithmetics_model.rock.permx[3:6, 3:6, 1:1] + 5)

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
