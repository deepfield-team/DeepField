FROM python:3.11

RUN pip install -U pip
RUN pip --no-cache-dir install jupyter

RUN git clone https://github.com/deepfield-team/DeepField.git /home/DeepField

WORKDIR "/home/DeepField"

RUN pip --no-cache-dir install . 

EXPOSE 8888

ENTRYPOINT ["/bin/bash"]
