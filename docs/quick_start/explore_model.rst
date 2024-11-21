=======================
Explore reservoir model
=======================

Once the reservoir model is loaded using

.. code-block:: python

	from deepfield import Field

	model = Field('/path/to/model.DATA').load()

one can explore the model data.

Reservoir model data is distributed between `Field` components:

* Grid
* Rock
* States
* Wells
* Tables
* Faults
* Aquifer

Each component can be accessed by its name, for example

.. code-block:: python

	model.grid

Components contain attributes corresponding to ECLIPSE keywords. For example, the keyword DIMES that contains reservoir grid dimensions can be accessed using

.. code-block:: python

	model.grid.dimens

Note that the syntax model.grid.dimens, model.grid.DIMENS, and model.grid['DIMENS'] are equivalent.

List of attributes can be obtained using

.. code-block:: python

	model.grid.attributes

Apart from the source data of the reservoir model, components have a number of computed attributes and specific methods. For example, one can get grid cell volumes using 

.. code-block:: python

	model.grid.cell_volumes

Read more about details of the components in the `tutorials <https://github.com/deepfield-team/DeepField/blob/main/tutorials>`_ provided in the DeepField repository.