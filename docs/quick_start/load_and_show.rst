=============================
Load and show reservoir model
=============================

`DeepField` allows reading reservoir models in ECLIPSE file format.
The main file of the model must have a `.DATA` extension.
For example, you can find a number of reservoir models in the `open_data` directory in the `DeepField` repository.

To load reservoir model data, use

.. code-block:: python

	from deepfield import Field

	model = Field('/path/to/model.DATA').load()

Now you can look at the model in 3d using the `show` method. For example, let's visualize the porosity distribution over the model:

.. code-block:: python

	model.show(attr='PORO')

This will open a separate window where you can rotate and zoom the model. The `show` method has a number of additional options for slicing, thresholding and controlling visual settings. These options are presented in the notebook 
`tutorials/07.Visualization.ipynb <https://github.com/deepfield-team/DeepField/blob/main/tutorials/07.Visualization.ipynb>`_.