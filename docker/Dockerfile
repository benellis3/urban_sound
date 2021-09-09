FROM pytorch/pytorch

COPY . /source

RUN conda config --add channels conda-forge && conda install obspy

RUN (cd /source/ && python setup.py develop)

WORKDIR /source/