#!/bin/bash

export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries/linux/mpi/intel64/lib
$1 --server-url $2 $3 -n 32 $4