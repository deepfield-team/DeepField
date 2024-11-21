====================
Dump reservoir model
====================

Reservoir model data can be saved in two formats:

* ECLIPSE
* HDF5

By using the ECLIPLE file format, you ensure file compatibility with standard reservoir modeling software. In this case, use

.. code-block:: python

	model.dump('path/where/to/save/new_model.DATA')

HDF5 is useful if you need to significantly speed up reading reservoir model data. For example, if you have many reservoir models that need to be processed in a loop. In this case, dump the model as follows:

.. code-block:: python

	model.dump('path/where/to/save/new_model.hdf5')

Reading the model remains standard:

.. code-block:: python

	new_model = Field('new_model.hdf5').load()