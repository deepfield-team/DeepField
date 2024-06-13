FROM python:3.11

RUN apt-get update && apt-get install -y \
		libgl1-mesa-glx \
		libhdf5-dev \
		libopenblas-dev \
		liblapack-dev \
		libssl-dev \
		&& \
	apt-get clean && \
	apt-get autoremove

RUN pip install -U pip
RUN pip --no-cache-dir install jupyter

RUN git clone https://github.com/deepfield-team/DeepField.git /home/DeepField

WORKDIR "/home/DeepField"

RUN pip --no-cache-dir install . 

EXPOSE 8888

ENTRYPOINT ["/bin/bash"]