Required: `docker`, `nvidia-docker2`

**All commands must be run from the parent (DeepField) folder** 

## Building docker image

*tag* - is name for docker image 

To build docker image run the command:

    docker build -t <tag> Dockerfile .

## Running docker container

*run_name* - is name for container born by image

#### For docker `version < 19.03`:

Run docker container without jupyter notebook access and no shared folders using:
* `docker run -it --rm --name=<run_name> <tag>`

Run docker container without jupyter notebook access but share a folder:
* `docker run -v <host dir>:<docker dir> -it --rm --name=run_name <tag>`

Run docker container with jupyter notebook access and shared folder:
* `docker run -p <host port>:8888 -v <host dir>:<docker dir> -it --rm --name=run_name <tag>`


#### For docker `version >= 19.03`:
Run docker container without jupyter notebook access and no shared folders using:
* `docker run -it --rm --name=<run_name> <tag>`

Run docker container without jupyter notebook access but share a folder:
* `docker run -v <host dir>:<docker dir> -it --rm --name=run_name <tag>`

Run docker container with jupyter notebook access and shared folder:
* `docker run -p <host port>:8888 -v <host dir>:<docker dir> -it --rm --name=run_name <tag>`

## Running Jupyter Notebook

To run jupyter notebook inside a container run:
    
    jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root
