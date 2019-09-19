FROM python:3.7-buster

RUN apt-get update && apt-get upgrade -y \
    && pip install pipenv
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
# For geopandas
RUN apt-get install -y libgdal-dev=2.4.0+dfsg-1+b1 libspatialindex-dev
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

ARG user
RUN adduser --disabled-password --gecos '' ${user}
USER ${user}
WORKDIR /home/${user}/dev
COPY jupyter_config.txt Pipfile Pipfile.lock ./
RUN pipenv sync --dev && \
    pipenv run jupyter notebook --generate-config && \
    cat jupyter_config.txt >> ~/.jupyter/jupyter_notebook_config.py
# The shape file for the landuse is obtained from https://data.gov.sg/dataset/master-plan-2014-land-use?resource_id=0d1a6cda-7cad-4b17-b9a8-9e173afebbc1
