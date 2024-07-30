[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)


# DeepField

Python framework for reservoir engineering.

![img](static/3d_basic.PNG)

## Features

* reservoir representation with Grid, Rock, States, Wells, Aquifer and PVT-tables components
* interactive 3D visualization
* reservoir preprocessing tools
* detailed [documentation](https://deepfield-team.github.io/DeepField/)
* [tutorials](/tutorials) to explore the framework step-by-step


## Installation

Clone the repository:

    git clone https://github.com/deepfield-team/DeepField.git

Working with a remote server, it is recommended to install
VNC for remote rendering of 3D graphics (follow this [instruction](./vnc/README.md))

Another option is to build the docker image with DeepField inside.
Instructions and dockerfile are provided in the [docker](./docker) directory.

```
Note: the project is in developement. We welcome contributions and collaborations.
```

## Quick start

Load a reservoir model from `.DATA` file:

```python

  from deepfield import Field

  model = Field('model.data').load()
```

See the [tutorials](./tutorials) to explore the framework step-by-step
and the [documentation](https://deepfield-team.github.io/DeepField/) for more details.


## Model formats

Initial reservoir model can be given in a mixture of ECLIPSE, MORE, PETREL, tNavigator formats.
However, there is no guarantee that any mixture will be understood.
Main file should be in .DATA file. Dependencies can be text and binary files including common formats:

* .GRDECL
* .INC
* .RSM
* .UNRST
* .RSSPEC
* .UNSMRY
* .SMSPEC
* .EGRID
* .INIT

Within the `DeepField` framework it is recommended to use the HDF5 format
to speed up data load and dump in Python-friendly data formats. In this
case all data are contained in a single .HDF5 file. At any point the model
can be exported back into .DATA text and binary files to ensure a compatibility
with conventional software.
